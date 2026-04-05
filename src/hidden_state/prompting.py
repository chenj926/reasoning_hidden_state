from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

DEFAULT_SYSTEM_PROMPT = "You are a careful mathematical reasoning assistant."

GSM8K_COT_TEMPLATE = """Solve the following math problem step by step.
The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

NORMAL_TEMPLATE = """Answer the following math problem.
The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.

Answer:"""

MATH_STOCK_TEMPLATE = "Question: {prompt}\nAnswer:"
MATH_GREEDY_TEMPLATE = GSM8K_COT_TEMPLATE


@dataclass
class PromptPair:
    question: str
    cot_prompt: str
    norm_prompt: str


def maybe_apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    *,
    use_chat_template: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    if not use_chat_template:
        return prompt_text
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_pair_for_question(question: str, benchmark_style: str) -> PromptPair:
    if benchmark_style == "gsm8k":
        cot_prompt = GSM8K_COT_TEMPLATE.format(prompt=question)
        norm_prompt = NORMAL_TEMPLATE.format(prompt=question)
    elif benchmark_style == "math_stock":
        cot_prompt = GSM8K_COT_TEMPLATE.format(prompt=question)
        norm_prompt = MATH_STOCK_TEMPLATE.format(prompt=question)
    elif benchmark_style == "math_greedy":
        cot_prompt = MATH_GREEDY_TEMPLATE.format(prompt=question)
        norm_prompt = NORMAL_TEMPLATE.format(prompt=question)
    else:
        cot_prompt = GSM8K_COT_TEMPLATE.format(prompt=question)
        norm_prompt = NORMAL_TEMPLATE.format(prompt=question)
    return PromptPair(question=question, cot_prompt=cot_prompt, norm_prompt=norm_prompt)


def infer_raw_question_from_query(query: str) -> str:
    if "Solve the following math problem step by step." in query:
        text = query
        marker_a = "problem.\n\n"
        marker_b = "\n\nRemember to put your answer"
        if marker_a in text and marker_b in text:
            start = text.index(marker_a) + len(marker_a)
            end = text.index(marker_b, start)
            return text[start:end].strip()

    m = re.match(r"Question:\s*(.*)\nAnswer:\s*$", query, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return query.strip()
