from __future__ import annotations

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import math_normalizer
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


def prompt_fn(line: dict, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['problem']}\nAnswer:",
        choices=[f" {line['solution']}"],
        gold_index=0,
        specific={
            'raw_question': line['problem'],
            'benchmark_style': 'math_stock',
        },
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f'math_stock_semantics:{subject}',
        prompt_function=prompt_fn,
        hf_repo='DigitalLearningGmbH/MATH-lighteval',
        hf_subset=subject,
        hf_avail_splits=['train', 'test'],
        evaluation_splits=['test'],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=2048,
        stop_sequence=['\n'],
        metrics=[Metrics.maj_at_k],
        version=1,
    )
    for subject in SUBJECTS
]
