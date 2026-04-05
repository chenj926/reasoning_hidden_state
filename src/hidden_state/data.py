from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from datasets import load_dataset


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass
class BenchmarkRecord:
    record_id: str
    benchmark: str
    question: str
    gold_answer: str
    metadata: dict


def _limit_list(items: list, max_samples: Optional[int]) -> list:
    if max_samples is None:
        return items
    return items[: min(len(items), max_samples)]


def load_gsm8k_records(split: str = "test", max_samples: Optional[int] = None) -> list[BenchmarkRecord]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    records: list[BenchmarkRecord] = []
    for idx, row in enumerate(dataset):
        gold = row["answer"].split("####")[-1].strip()
        records.append(
            BenchmarkRecord(
                record_id=f"gsm8k-{split}-{idx}",
                benchmark="gsm8k",
                question=row["question"],
                gold_answer=gold,
                metadata={},
            )
        )
    return _limit_list(records, max_samples)


def load_math_records(
    split: str = "test",
    subjects: Optional[list[str]] = None,
    max_samples_per_subject: Optional[int] = None,
) -> list[BenchmarkRecord]:
    if subjects is None:
        subjects = list(MATH_SUBJECTS)

    records: list[BenchmarkRecord] = []
    for subject in subjects:
        dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", subject, split=split)
        local_records: list[BenchmarkRecord] = []
        for idx, row in enumerate(dataset):
            local_records.append(
                BenchmarkRecord(
                    record_id=f"math-{subject}-{split}-{idx}",
                    benchmark="math",
                    question=row["problem"],
                    gold_answer=row["solution"],
                    metadata={"subject": subject},
                )
            )
        records.extend(_limit_list(local_records, max_samples_per_subject))
    return records


def load_tgs_auxiliary_questions_from_math_train(
    count: int,
    *,
    offset: int = 0,
    subjects: Optional[list[str]] = None,
) -> list[str]:
    records = load_math_records(
        split="train",
        subjects=subjects,
        max_samples_per_subject=None,
    )
    selected = records[offset : offset + count]
    return [record.question for record in selected]


def load_tgs_auxiliary_questions_from_gsm8k_train(
    count: int,
    *,
    offset: int = 0,
) -> list[str]:
    records = load_gsm8k_records(split="train", max_samples=None)
    selected = records[offset : offset + count]
    return [record.question for record in selected]
