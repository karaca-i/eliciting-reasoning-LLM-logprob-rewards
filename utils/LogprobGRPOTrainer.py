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

class LogprobGRPOTrainer(GRPOTrainer):
    """
    A GRPOTrainer variant that back-propagates through `i + r + a_g`
    (gold answer) instead of the default `i + r + a_p`.
    
    The dataset **must** contain an `"answer"` column (the ground-truth
    answer text without XML tags).
    """

    def _generate_and_score_completions(  # noqa: C901  (we purposefully keep the same name/signature)
        self, inputs: List[dict[str, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        # Step 1 – run the original pipeline (uses a_p)
        out = super()._generate_and_score_completions(inputs)

        # Nothing to change for reward/advantages – they’re already inside `out`.
        # We only replace the tensors used later by `compute_loss`.
        device = self.accelerator.device
        tokenizer = self.processing_class

        # Recover the *reasoning* part that the model just generated.
        predicted_texts = tokenizer.batch_decode(
            out["completion_ids"], skip_special_tokens=True
        )

        gold_answers = [example["answer"] for example in inputs]
        new_completion_texts = []

        for pred, gold in zip(predicted_texts, gold_answers):
            # Split at first "<answer>" (if present) to keep reasoning
            if "<answer>" in pred:
                reasoning_prefix = pred.split("<answer>")[0]
                new_completion = (
                    f"{reasoning_prefix}<answer>\n{gold.strip()}\n</answer>"
                )
            else:
                # Fallback – model forgot the tag: treat whole output as reasoning
                new_completion = (
                    f"{pred.rstrip()}\n<answer>\n{gold.strip()}\n</answer>"
                )
            new_completion_texts.append(new_completion)

        # Tokenise the new gold-answer completions
        comp_inputs = tokenizer(
            new_completion_texts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        ).to(device)

        new_completion_ids = comp_inputs["input_ids"]                   # (B, L_c)
        new_completion_mask = comp_inputs["attention_mask"].int()       # (B, L_c)

        # Replace p-a_p tensors with p-a_g tensors
        out["completion_ids"] = new_completion_ids
        out["completion_mask"] = new_completion_mask
        # Force importance-sampling baseline to current model log-probs
        out["old_per_token_logps"] = None

        return out