from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from functools import partial
import math
from trl.trainer.utils import selective_log_softmax
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

load_name_prefix = "outputs_logprob_scale_2"
load_name_prefix2 = "outputs_logprob_scale_5"
load_name_prefix3 = "outputs_no_exact_match"

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

model_name = "Qwen/Qwen2.5-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
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

# GET TEST DATASET
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
]

for checkpoint in checkpoints:
    load_name = f"{load_name_prefix}/{checkpoint}"
    load_name2 = f"{load_name_prefix2}/{checkpoint}"
    load_name3 = f"{load_name_prefix3}/{checkpoint}"

    total = 0
    correct = 0
    correct2 = 0
    correct3 = 0
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

        output2 = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(load_name2),
        )[0].outputs[0].text

        output3 = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(load_name3),
        )[0].outputs[0].text

        pred = extract_xml_answer(output)
        pred2 = extract_xml_answer(output2)
        pred3 = extract_xml_answer(output3)
        gold = example["answer"]

        if pred.strip() == gold.strip():
            correct += 1
        if pred2.strip() == gold.strip():
            correct2 += 1
        if pred3.strip() == gold.strip():
            correct3 += 1
        total += 1

    printc(f'Accuracy {load_name}: {correct}/{total} = {correct/total}', color='red')
    printc(f'Accuracy {load_name2}: {correct2}/{total} = {correct2/total}', color='green')
    printc(f'Accuracy {load_name3}: {correct3}/{total} = {correct3/total}', color='blue')
    with open('final_accuracy.txt', 'a') as f:
        f.write(f'--------------------------------\n')
        f.write(f'Accuracy {load_name}: {correct}/{total} = {correct/total}\n')
        f.write(f'Accuracy {load_name2}: {correct2}/{total} = {correct2/total}\n')
        f.write(f'Accuracy {load_name3}: {correct3}/{total} = {correct3/total}\n')