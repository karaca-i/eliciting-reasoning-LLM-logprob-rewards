# Eliciting Reasoning LLM Logprob Rewards

## Group Name

RL Barcelona

## Group Members

- Ibrahim Karaca
- Orhun Ege Celik
- Semih Zaman
- Arkadiusz Rozmarynowicz
- David Grishchuk

This repository explores methods for eliciting reasoning through log-probability-based rewards in Large Language Models (LLMs).

## Overview

This project investigates how log-probability-based reward signals can be used to elicit and improve reasoning abilities in Large Language Models (LLMs). We focus on constructing reward functions based on the model’s output logprobs, aiming to guide the model toward more interpretable, step-by-step reasoning in mathematical and logical tasks.

### Model and Methods

- **Base Model**: We primarily use [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) as the backbone LLM, interfaced via the [Unsloth](https://github.com/unslothai/unsloth) library’s `FastLanguageModel` for efficient loading and inference.
- **Parameter-Efficient Fine-Tuning (LoRA)**: We leverage Low-Rank Adaptation (LoRA) to fine-tune the model on our tasks. LoRA is applied to key transformer components (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with customizable rank and alpha parameters. This allows us to efficiently adapt the model’s reasoning patterns with a small memory footprint.
- **Training**: We use the GRPOTrainer from the `trl` library, with custom reward functions based on logprobs. This setup enables us to train the model using reward signals that reflect the quality of reasoning, not just the final answer.
- **Datasets**: The main dataset used is [GSM8K](https://huggingface.co/datasets/openai/gsm8k), a collection of grade-school math problems, where the model is prompted to generate both the reasoning steps and the final answer in a structured format.
- **Prompting**: We use a system prompt that enforces the following output structure:
  ```
  <reasoning>
  ...step-by-step reasoning...
  </reasoning>
  <answer>
  ...final answer...
  </answer>
  ```
  This helps in evaluating and rewarding not just the correctness of the answer, but also the reasoning process.

### Goals

- **Elicit Interpretable Reasoning**: By tying rewards to logprobs over intermediate reasoning steps, we encourage the model to reason transparently, not just output the correct answer.
- **Efficient Adaptation**: Using LoRA and Unsloth’s efficient model backend, we keep hardware requirements accessible while enabling rapid experimentation.
- **Flexible Evaluation**: We provide scripts for both training and evaluation, allowing comparison between different reward schemes and LoRA configurations.

---

## Features

- Experiments with logprob-based reward shaping
- Evaluation of reasoning abilities in LLMs
- Python-based codebase

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/karaca-i/eliciting-reasoning-LLM-logprob-rewards.git
   cd eliciting-reasoning-LLM-logprob-rewards
   ```

2. **(Optional) Install dependencies**

   ```bash
   pip install unsloth vllm
   ```
   or, if you are using Google Colab:
   ```bash
   pip install --no-deps unsloth vllm==0.8.5.post1
   ```
3. **Run experiments**

    There are 3 experiments that can be run in this repo. Run the following to obtain log-probability reward based LoRA checkpoints.
    ```bash
    pip install unsloth vllm
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
