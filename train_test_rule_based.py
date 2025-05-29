from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from functools import partial
import math
from trl.trainer.utils import selective_log_softmax

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

save_name = "saved_1_400_exact_match"

def printc(text, color="red"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m"
    }
    end_color = "\033[0m"
    
    color_code = colors.get(color.lower(), colors["red"])
    print(f"{color_code}{text}{end_color}")

old_model_name = "Qwen/Qwen2.5-3B-Instruct"
new_model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
new_model_name = old_model_name

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = new_model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# reasoning_start = "<reasoning>"
# reasoning_end = "</reasoning>"
# solution_start = "<answer>"
# solution_end = "</answer>"

# SYSTEM_PROMPT = f"""
# You are given a problem.
# Think about the problem and place your reasoning between {reasoning_start} and {reasoning_end}.
# Then, provide your numerical answer between {solution_start} and {solution_end}
# """


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")

    # with open('responses.txt', 'a') as f:
    #     f.write(f"Question:\n{q}\n")
    #     f.write(f"Answer:\n{answer[0]}\n")
    #     f.write(f"Response:\n{responses[0]}\n")
    #     f.write(f"Extracted-answer:\n{extracted_responses[0]}\n\n")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

@torch.no_grad()
def logprob_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    For each prompt+generated chain-of-thought, reward = average log-prob of the gold answer
    given that chain. Implements R = log P(a_gold | p, r) via selective_log_softmax.
    """
    device = model.device
    rewards: list[float] = []
    for prompt, completion, a_gold in zip(prompts, completions, answer):
        # 1) reconstruct the full prompt string
        prompt_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        # 2) grab the raw generated content (it’s wrapped in a list-of-dicts)
        gen = completion[0]["content"]
        # 3) split off just the reasoning (everything before your <answer> tag)
        reasoning = gen.split("<answer>")[0]
        # 4) build two token-sequences: context = p + reasoning, full = context + gold_answer
        ctx_str  = prompt_text + reasoning
        full_str = ctx_str + a_gold.strip()

        ctx_ids  = tokenizer(ctx_str,  return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
        full_ids = tokenizer(full_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]

        # 5) run the model once over full_ids to get logits
        outputs = model(input_ids=full_ids.unsqueeze(0))
        logits  = outputs.logits[0]            # shape [seq_len, vocab_size]

        # 6) isolate just the rows that predicted the gold-answer tokens
        start        = ctx_ids.shape[0] - 1    # last context logit
        answer_logits = logits[start:-1]       # one row per answer token
        answer_ids    = full_ids[start+1:]     # the gold-answer token IDs

        # 7) compute per-token log-probs, memory-efficiently
        per_tok = selective_log_softmax(
            answer_logits.unsqueeze(0),          # [1, answer_len, vocab]
            answer_ids.unsqueeze(0)              # [1, answer_len]
        ).squeeze(0)                             # [answer_len]

        # 8) average to get a single scalar reward
        rewards.append(per_tok.sum().item() / answer_ids.size(0))

    return rewards



from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1500,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

# logprob_reward = partial(logprob_reward_func, model=model, tokenizer=tokenizer)
# logprob_reward.__name__ = "logprob_reward_func"

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        # logprob_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
    output_dir = "outputs_old/outputs_exact_match_final_kesin",
)
latest = None # Set to None to start from scratch
trainer.train(resume_from_checkpoint = latest) # Set to "latest" to resume training


question_ = "How many r's are in strawberry?"
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : question_},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = max_seq_length,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

printc(output)

"""And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""

#model.save_lora(save_name)

"""Now we load the LoRA and test:"""

text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : question_},
], tokenize = False, add_generation_prompt = True)

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = max_seq_length,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora(save_name),
)[0].outputs[0].text

printc(output)

dataset = get_gsm8k_questions("test")
printc(len(dataset))
dataset = dataset.select(range(100))  # type: ignore


checkpoints = [
    'checkpoint-100',
    'checkpoint-200',
    'checkpoint-300',
    'checkpoint-400',
    'checkpoint-500',
    'checkpoint-600',
    'checkpoint-700',
    'checkpoint-800',
    'checkpoint-900',
    'checkpoint-1000',
    'checkpoint-1100',
    'checkpoint-1200',
    'checkpoint-1300',
    'checkpoint-1400',
    'checkpoint-1500',
]

for checkpoint in checkpoints:
    load_name = f"outputs_old/outputs_exact_match_final_kesin/{checkpoint}"

    total = 0
    correct = 0
    from tqdm import tqdm

    for example in tqdm(dataset):
        question_ = example['prompt'][1]['content']
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_},
        ], tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )

        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(load_name),
        )[0].outputs[0].text

        pred = extract_xml_answer(output)
        gold = example["answer"]

        if pred.strip() == gold.strip():
            correct += 1
        total += 1

    printc(f'Accuracy {load_name}: {correct}/{total} = {correct/total}', color='red')
    with open('final_accuracy.txt', 'a') as f:
        f.write(f'--------------------------------\n')
        f.write(f'Accuracy {load_name}: {correct}/{total} = {correct/total}\n')