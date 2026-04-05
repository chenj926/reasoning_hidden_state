from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from .data import BenchmarkRecord
from .extraction import build_vanilla_steering_bundle
from .generation import generate_text
from .parsing import answers_equivalent, extract_final_answer
from .prompting import build_prompt_pair
from .steering import SteeringBundle, select_last_k_layers
from .utils import ensure_dir, json_dump, jsonl_dump


@dataclass
class ExampleResult:
    record_id: str
    benchmark: str
    question: str
    gold_answer_raw: str
    gold_answer_extracted: Optional[str]
    prediction_raw: str
    prediction_extracted: Optional[str]
    is_correct: bool
    method: str
    alpha: float
    k: int
    norm_preserving: bool
    layers: list[int]
    metadata: dict


def _extract_gold_answer(record: BenchmarkRecord) -> str:
    if record.benchmark == "gsm8k":
        return record.gold_answer
    extracted = extract_final_answer(record.gold_answer)
    return extracted if extracted is not None else record.gold_answer


def evaluate_records(
    *,
    records: list[BenchmarkRecord],
    loaded_model,
    method: str,
    k: int,
    alpha: float,
    norm_preserving: bool,
    max_new_tokens: int,
    output_dir: Optional[str] = None,
    tgs_full_bundle: Optional[SteeringBundle] = None,
    use_chat_template: bool = True,
    generation_prompt_kind: str = "cot",
    progress_desc: str = "Evaluating",
) -> dict:
    if method not in {"baseline", "baseline_normal", "vanilla", "tgs"}:
        raise ValueError(f"Unsupported method: {method}")

    if method == "tgs" and tgs_full_bundle is None:
        raise ValueError("tgs_full_bundle must be provided when method='tgs'.")

    selected_tgs_bundle = None
    if method == "tgs":
        selected_tgs_bundle = select_last_k_layers(tgs_full_bundle, k)

    example_results: list[ExampleResult] = []

    for record in tqdm(records, desc=progress_desc):
        prompts = build_prompt_pair(
            loaded_model.tokenizer,
            record.question,
            use_chat_template=use_chat_template,
        )
        if generation_prompt_kind == "cot":
            generation_prompt = prompts.cot_model_prompt
        elif generation_prompt_kind == "norm":
            generation_prompt = prompts.norm_model_prompt
        else:
            raise ValueError(
                "generation_prompt_kind must be either 'cot' or 'norm'."
            )

        steering_bundle = None
        if method == "baseline":
            steering_bundle = None
        elif method == "baseline_normal":
            steering_bundle = None
            generation_prompt = prompts.norm_model_prompt
        elif method == "vanilla":
            vanilla_full_bundle, _ = build_vanilla_steering_bundle(
                loaded_model,
                record.question,
                use_chat_template=use_chat_template,
            )
            steering_bundle = select_last_k_layers(vanilla_full_bundle, k)
        elif method == "tgs":
            steering_bundle = selected_tgs_bundle

        prediction_raw = generate_text(
            loaded_model,
            generation_prompt,
            steering_bundle=steering_bundle,
            alpha=alpha,
            norm_preserving=norm_preserving,
            max_new_tokens=max_new_tokens,
        )
        prediction_extracted = extract_final_answer(prediction_raw)
        gold_answer_extracted = _extract_gold_answer(record)
        is_correct = answers_equivalent(prediction_extracted, gold_answer_extracted)

        layers = steering_bundle.layer_indices if steering_bundle is not None else []

        example_results.append(
            ExampleResult(
                record_id=record.record_id,
                benchmark=record.benchmark,
                question=record.question,
                gold_answer_raw=record.gold_answer,
                gold_answer_extracted=gold_answer_extracted,
                prediction_raw=prediction_raw,
                prediction_extracted=prediction_extracted,
                is_correct=is_correct,
                method=method,
                alpha=alpha,
                k=k,
                norm_preserving=norm_preserving,
                layers=layers,
                metadata=record.metadata,
            )
        )

    accuracy = (
        sum(int(item.is_correct) for item in example_results) / len(example_results)
        if example_results
        else 0.0
    )

    metrics = {
        "num_examples": len(example_results),
        "num_correct": sum(int(item.is_correct) for item in example_results),
        "accuracy": accuracy,
    }

    if records and records[0].benchmark == "math":
        per_subject: dict[str, dict[str, float | int]] = {}
        for item in example_results:
            subject = str(item.metadata.get("subject", "unknown"))
            if subject not in per_subject:
                per_subject[subject] = {"num_examples": 0, "num_correct": 0}
            per_subject[subject]["num_examples"] += 1
            per_subject[subject]["num_correct"] += int(item.is_correct)
        for subject, subject_stats in per_subject.items():
            subject_stats["accuracy"] = (
                subject_stats["num_correct"] / subject_stats["num_examples"]
                if subject_stats["num_examples"] > 0
                else 0.0
            )
        metrics["per_subject"] = per_subject

    output = {
        "metrics": metrics,
        "examples": [asdict(item) for item in example_results],
    }

    if output_dir is not None:
        out_dir = ensure_dir(output_dir)
        json_dump(metrics, out_dir / "scores.json")
        jsonl_dump([asdict(item) for item in example_results], out_dir / "predictions.jsonl")

    return output
