from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _first_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_prediction_record(task_name: str, detail: dict[str, Any], sample_index: int) -> dict[str, Any]:
    doc = detail.get("doc", {}) or {}
    model_response = detail.get("model_response", {}) or {}
    metric = detail.get("metric", {}) or {}
    specific = doc.get("specific", {}) or {}

    predictions = _safe_list(model_response.get("text"))
    predictions_post_processed = _safe_list(model_response.get("text_post_processed"))
    gold_choices = _safe_list(doc.get("choices"))
    output_tokens = _safe_list(model_response.get("output_tokens"))

    return {
        "sample_index": int(sample_index),
        "task_name": task_name,
        "doc_id": doc.get("id"),
        "benchmark_style": specific.get("benchmark_style"),
        "raw_question": specific.get("raw_question"),
        "query": doc.get("query"),
        "model_input": model_response.get("input"),
        "gold_choices": gold_choices,
        "gold_choice": _first_or_none(gold_choices),
        "extracted_golds": _safe_list(specific.get("extracted_golds")),
        "predictions": predictions,
        "prediction": _first_or_none(predictions),
        "predictions_post_processed": predictions_post_processed,
        "prediction_post_processed": _first_or_none(predictions_post_processed),
        "extracted_predictions": _safe_list(specific.get("extracted_predictions")),
        "metric": metric,
        "is_correct": bool(metric.get("extractive_match") == 1.0),
        "stop_sequences": _safe_list(doc.get("stop_sequences")),
        "output_token_count": len(_first_or_none(output_tokens) or []),
    }


def write_predictions_jsonl(
    details_by_task: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path,
    model_name: str,
    date_id: str,
) -> list[Path]:
    base_dir = Path(output_dir) / "predictions" / model_name.strip("/") / date_id
    base_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for task_name, task_details in sorted(details_by_task.items()):
        output_path = base_dir / f"predictions_{task_name}_{date_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for sample_index, detail in enumerate(task_details):
                record = build_prediction_record(task_name, detail, sample_index)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_files.append(output_path)
    return written_files


def latest_saved_date_id(output_dir: str | Path, model_name: str) -> str | None:
    details_dir = Path(output_dir) / "details" / model_name.strip("/")
    if not details_dir.exists():
        return None
    date_ids = sorted(path.name for path in details_dir.iterdir() if path.is_dir())
    return date_ids[-1] if date_ids else None
