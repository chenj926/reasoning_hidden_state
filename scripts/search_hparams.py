#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.hidden_state.data import (
    load_gsm8k_records,
    load_math_records,
)
from src.hidden_state.evaluate import evaluate_records
from src.hidden_state.modeling import load_model_and_tokenizer
from src.hidden_state.steering import load_steering_bundle
from src.hidden_state.utils import ensure_dir, json_dump, jsonl_dump, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["gsm8k", "math"])
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--method", required=True, choices=["vanilla", "tgs"])
    parser.add_argument("--k-grid", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--math-subjects", nargs="*", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--norm-preserving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tgs-vector-path", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    loaded_model = load_model_and_tokenizer(args.model, precision=args.precision)

    if args.benchmark == "gsm8k":
        records = load_gsm8k_records(split="train", max_samples=args.max_samples)
        max_new_tokens = args.max_new_tokens or 256
    else:
        records = load_math_records(
            split="train",
            subjects=args.math_subjects,
            max_samples_per_subject=args.max_samples,
        )
        max_new_tokens = args.max_new_tokens or 768

    tgs_bundle = None
    if args.method == "tgs":
        if not args.tgs_vector_path:
            raise ValueError("--tgs-vector-path is required when --method tgs")
        tgs_bundle = load_steering_bundle(args.tgs_vector_path)

    grid_results = []
    best = None

    for k, alpha in itertools.product(args.k_grid, args.alpha_grid):
        trial_out_dir = output_dir / f"k_{k}_alpha_{alpha}"
        results = evaluate_records(
            records=records,
            loaded_model=loaded_model,
            method=args.method,
            k=k,
            alpha=alpha,
            norm_preserving=args.norm_preserving,
            max_new_tokens=max_new_tokens,
            output_dir=str(trial_out_dir),
            tgs_full_bundle=tgs_bundle,
            use_chat_template=True,
            generation_prompt_kind="cot",
            progress_desc=f"Search {args.method} k={k} alpha={alpha}",
        )
        row = {
            "benchmark": args.benchmark,
            "method": args.method,
            "k": k,
            "alpha": alpha,
            "accuracy": results["metrics"]["accuracy"],
            "num_examples": results["metrics"]["num_examples"],
            "num_correct": results["metrics"]["num_correct"],
        }
        grid_results.append(row)
        if best is None or row["accuracy"] > best["accuracy"]:
            best = row

    json_dump(
        {
            "model": args.model,
            "precision": args.precision,
            "benchmark": args.benchmark,
            "method": args.method,
            "k_grid": args.k_grid,
            "alpha_grid": args.alpha_grid,
            "max_samples": args.max_samples,
            "norm_preserving": args.norm_preserving,
            "best": best,
        },
        output_dir / "search_summary.json",
    )
    jsonl_dump(grid_results, output_dir / "search_grid.jsonl")
    print("Best configuration:", best)


if __name__ == "__main__":
    main()
