from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from functools import partial
import math
from trl.trainer.utils import selective_log_softmax
from trl import GRPOConfig, GRPOTrainer
from torch.nn import functional as F
import re
from datasets import load_dataset, Dataset
from typing import Any, Union, Optional, List
from vllm import SamplingParams

from utils.utils import *

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

save_name = "saved_1_1000_step_rule_based"
model_name = "Qwen/Qwen2.5-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
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

dataset = get_gsm8k_questions()

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
    max_steps = 1000,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()

question_ = "How many r's are in strawberry?"
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : question_},
], tokenize = False, add_generation_prompt = True)

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

"""And now with the LoRA we just trained with GRPO - we first save the LoRA first"""

model.save_lora(save_name)

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