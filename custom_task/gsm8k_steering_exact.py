from __future__ import annotations

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:""".strip()


def prompt_fn(line: dict, task_name: str = None):
    gold = line['answer'].split('####')[-1].strip()
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line['question']),
        choices=[f" {gold}"],
        gold_index=0,
        specific={
            'raw_question': line['question'],
            'benchmark_style': 'gsm8k',
        },
    )


gsm8k_steering_exact = LightevalTaskConfig(
    name='gsm8k_steering_exact',
    prompt_function=prompt_fn,
    hf_repo='openai/gsm8k',
    hf_subset='main',
    hf_avail_splits=['train', 'test'],
    evaluation_splits=['test'],
    few_shots_split=None,
    few_shots_select='random_sampling_from_train',
    generation_size=256,
    stop_sequence=['Question:'],
    metrics=[Metrics.expr_gold_metric],
    version=0,
)

TASKS_TABLE = [gsm8k_steering_exact]
