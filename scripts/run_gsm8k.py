#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hidden_state.data import load_gsm8k_records
from src.hidden_state.evaluate import evaluate_records
from src.hidden_state.modeling import load_model_and_tokenizer
from src.hidden_state.steering import load_steering_bundle
from src.hidden_state.utils import json_dump, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--method", default="baseline", choices=["baseline", "baseline_normal", "vanilla", "tgs"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--norm-preserving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tgs-vector-path", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    loaded_model = load_model_and_tokenizer(args.model, precision=args.precision)
    records = load_gsm8k_records(split=args.split, max_samples=args.max_samples)

    tgs_bundle = None
    if args.method == "tgs":
        if not args.tgs_vector_path:
            raise ValueError("--tgs-vector-path is required when --method tgs")
        tgs_bundle = load_steering_bundle(args.tgs_vector_path)

    results = evaluate_records(
        records=records,
        loaded_model=loaded_model,
        method=args.method,
        k=args.k,
        alpha=args.alpha,
        norm_preserving=args.norm_preserving,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        tgs_full_bundle=tgs_bundle,
        use_chat_template=True,
        generation_prompt_kind="cot",
        progress_desc="GSM8K evaluation",
    )

    json_dump(
        {
            "model": args.model,
            "precision": args.precision,
            "method": args.method,
            "k": args.k,
            "alpha": args.alpha,
            "norm_preserving": args.norm_preserving,
            "max_new_tokens": args.max_new_tokens,
            "split": args.split,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "tgs_vector_path": args.tgs_vector_path,
            "metrics": results["metrics"],
        },
        Path(args.output_dir) / "run_config.json",
    )

    print(results["metrics"])


if __name__ == "__main__":
    main()
