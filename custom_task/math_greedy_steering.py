from __future__ import annotations

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

SUBJECTS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus',
]

MATH_GREEDY_TEMPLATE = """Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:""".strip()


def prompt_fn(line: dict, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_GREEDY_TEMPLATE.format(prompt=line['problem']),
        choices=[f" {line['solution']}"],
        gold_index=0,
        specific={
            'raw_question': line['problem'],
            'benchmark_style': 'math_greedy',
        },
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f'math_greedy_steering:{subject}',
        prompt_function=prompt_fn,
        hf_repo='DigitalLearningGmbH/MATH-lighteval',
        hf_subset=subject,
        hf_avail_splits=['train', 'test'],
        evaluation_splits=['test'],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=768,
        stop_sequence=['Question:'],
        metrics=[Metrics.expr_gold_metric],
        version=0,
    )
    for subject in SUBJECTS
]
