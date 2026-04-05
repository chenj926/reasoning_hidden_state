from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

DEFAULT_SYSTEM_PROMPT = "You are a careful mathematical reasoning assistant."

# This CoT template intentionally mirrors the official Lighteval GSM8K task wording.
COT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your\n"
    "response should be of the form \"ANSWER: $ANSWER\" (without quotes)\n"
    "where $ANSWER is the answer to the problem.\n\n"
    "{prompt}\n\n"
    "Remember to put your answer on its own line at the end in the form\n"
    "\"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to\n"
    "the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)

NORMAL_TEMPLATE = (
    "Answer the following math problem. The last line of your\n"
    "response should be of the form \"ANSWER: $ANSWER\" (without quotes)\n"
    "where $ANSWER is the answer to the problem.\n\n"
    "{prompt}\n\n"
    "Remember to put your answer on its own line at the end in the form\n"
    "\"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to\n"
    "the problem, and you do not need to use a \\boxed command.\n\n"
    "Answer:"
)


@dataclass
class PromptPair:
    question: str
    cot_user_prompt: str
    norm_user_prompt: str
    cot_model_prompt: str
    norm_model_prompt: str


def _wrap_with_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_pair(
    tokenizer: AutoTokenizer,
    question: str,
    *,
    use_chat_template: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> PromptPair:
    cot_user_prompt = COT_TEMPLATE.format(prompt=question)
    norm_user_prompt = NORMAL_TEMPLATE.format(prompt=question)

    if use_chat_template:
        cot_model_prompt = _wrap_with_chat_template(tokenizer, cot_user_prompt, system_prompt)
        norm_model_prompt = _wrap_with_chat_template(tokenizer, norm_user_prompt, system_prompt)
    else:
        cot_model_prompt = cot_user_prompt
        norm_model_prompt = norm_user_prompt

    return PromptPair(
        question=question,
        cot_user_prompt=cot_user_prompt,
        norm_user_prompt=norm_user_prompt,
        cot_model_prompt=cot_model_prompt,
        norm_model_prompt=norm_model_prompt,
    )
