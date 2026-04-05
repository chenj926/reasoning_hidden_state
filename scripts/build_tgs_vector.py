#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hidden_state.data import (
    load_tgs_auxiliary_questions_from_gsm8k_train,
    load_tgs_auxiliary_questions_from_math_train,
)
from src.hidden_state.extraction import build_tgs_steering_bundle
from src.hidden_state.modeling import load_model_and_tokenizer
from src.hidden_state.steering import save_steering_bundle
from src.hidden_state.utils import json_dump, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--aux-source", default="math_train", choices=["math_train", "gsm8k_train"])
    parser.add_argument("--aux-count", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.aux_source == "math_train":
        questions = load_tgs_auxiliary_questions_from_math_train(
            count=args.aux_count,
            offset=args.offset,
        )
    else:
        questions = load_tgs_auxiliary_questions_from_gsm8k_train(
            count=args.aux_count,
            offset=args.offset,
        )

    bundle = load_model_and_tokenizer(args.model, precision=args.precision)
    steering_bundle = build_tgs_steering_bundle(bundle, questions)
    save_steering_bundle(steering_bundle, args.output)

    metadata_path = str(Path(args.output).with_suffix(".json"))
    json_dump(
        {
            "model": args.model,
            "precision": args.precision,
            "aux_source": args.aux_source,
            "aux_count": args.aux_count,
            "offset": args.offset,
            "seed": args.seed,
            "layer_indices": steering_bundle.layer_indices,
            "vector_norms": steering_bundle.vector_norms(),
            "bundle_source": steering_bundle.source,
            "bundle_metadata": steering_bundle.metadata,
        },
        metadata_path,
    )

    print(f"Saved steering bundle to: {args.output}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
