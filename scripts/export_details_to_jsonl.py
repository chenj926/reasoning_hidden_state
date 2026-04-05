#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hidden_state.predictions_export import build_prediction_record


def infer_task_name(parquet_path: Path) -> str:
    stem = parquet_path.stem
    if stem.startswith("details_"):
        stem = stem[len("details_") :]
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-parquet", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    parquet_path = Path(args.details_parquet).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Details parquet not found: {parquet_path}")

    task_name = infer_task_name(parquet_path)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else parquet_path.with_name(parquet_path.stem.replace("details_", "predictions_") + ".jsonl")
    )

    rows = pq.read_table(parquet_path).to_pylist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_index, row in enumerate(rows):
            record = build_prediction_record(task_name, row, sample_index)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(output_path)


if __name__ == "__main__":
    main()
