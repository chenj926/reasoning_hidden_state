#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.hidden_state.extraction import build_tgs_steering_bundle, build_vanilla_steering_bundle
from src.hidden_state.generation import generate_text
from src.hidden_state.modeling import load_model_and_tokenizer
from src.hidden_state.prompting import build_prompt_pair
from src.hidden_state.steering import load_steering_bundle, save_steering_bundle, select_last_k_layers
from src.hidden_state.utils import set_global_seed


def load_debug_questions(path: Path) -> list[str]:
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            questions.append(json.loads(line)["question"])
    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--method", default="baseline", choices=["baseline", "vanilla", "tgs"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug-prompts", default=str(PROJECT_ROOT / "data" / "debug_prompts.jsonl"))
    parser.add_argument("--save-tgs", default="")
    args = parser.parse_args()

    set_global_seed(args.seed)

    bundle = load_model_and_tokenizer(args.model, precision=args.precision)
    questions = load_debug_questions(Path(args.debug_prompts))

    steering_bundle = None
    if args.method == "tgs":
        full_bundle = build_tgs_steering_bundle(bundle, questions[: min(3, len(questions))])
        if args.save_tgs:
            save_steering_bundle(full_bundle, args.save_tgs)
        steering_bundle = select_last_k_layers(full_bundle, args.k)

    for idx, question in enumerate(questions):
        prompt_pair = build_prompt_pair(bundle.tokenizer, question, use_chat_template=True)
        if args.method == "baseline":
            text = generate_text(bundle, prompt_pair.cot_model_prompt, max_new_tokens=160)
        elif args.method == "vanilla":
            full_bundle, _ = build_vanilla_steering_bundle(bundle, question, use_chat_template=True)
            steering_bundle = select_last_k_layers(full_bundle, args.k)
            text = generate_text(
                bundle,
                prompt_pair.cot_model_prompt,
                steering_bundle=steering_bundle,
                alpha=args.alpha,
                norm_preserving=True,
                max_new_tokens=160,
            )
        else:
            text = generate_text(
                bundle,
                prompt_pair.cot_model_prompt,
                steering_bundle=steering_bundle,
                alpha=args.alpha,
                norm_preserving=True,
                max_new_tokens=160,
            )

        print("=" * 100)
        print(f"[{idx}] Question:\n{question}")
        print("-" * 100)
        print(text)


if __name__ == "__main__":
    main()
