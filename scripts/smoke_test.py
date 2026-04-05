#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hidden_state.extraction import build_vanilla_steering_bundle
from src.hidden_state.generation import generate_text, next_token_logits
from src.hidden_state.modeling import load_model_and_tokenizer
from src.hidden_state.prompting import build_prompt_pair
from src.hidden_state.steering import select_last_k_layers
from src.hidden_state.utils import detect_device_name, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_global_seed(args.seed)

    print("== Environment check ==")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", detect_device_name())
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    print("\n== Loading model ==")
    bundle = load_model_and_tokenizer(args.model, precision=args.precision)
    print("Loaded model:", bundle.model_id)
    print("Resolved device:", bundle.device)

    question = (
        "Jan has 3 boxes. Each box has 4 pencils. She buys 2 more pencils. "
        "How many pencils does she have now?"
    )

    print("\n== Building prompt pair ==")
    prompt_pair = build_prompt_pair(bundle.tokenizer, question, use_chat_template=True)
    print("CoT user prompt preview:\n", prompt_pair.cot_user_prompt[:240], "...")
    print("Normal user prompt preview:\n", prompt_pair.norm_user_prompt[:240], "...")

    print("\n== Extracting Vanilla steering bundle ==")
    full_bundle, _ = build_vanilla_steering_bundle(bundle, question, use_chat_template=True)
    selected_bundle = select_last_k_layers(full_bundle, args.k)
    print("Selected layers:", selected_bundle.layer_indices)
    print("Vector norms:", selected_bundle.vector_norms())

    print("\n== Comparing next-token logits ==")
    plain_logits = next_token_logits(bundle, prompt_pair.cot_model_prompt)
    steered_logits = next_token_logits(
        bundle,
        prompt_pair.cot_model_prompt,
        steering_bundle=selected_bundle,
        alpha=args.alpha,
        norm_preserving=True,
    )
    diff = (steered_logits - plain_logits).abs()
    print("Mean abs logit diff:", float(diff.mean().item()))
    print("Max abs logit diff:", float(diff.max().item()))

    print("\n== Plain generation ==")
    plain_text = generate_text(
        bundle,
        prompt_pair.cot_model_prompt,
        max_new_tokens=128,
    )
    print(plain_text)

    print("\n== Steered generation ==")
    steered_text = generate_text(
        bundle,
        prompt_pair.cot_model_prompt,
        steering_bundle=selected_bundle,
        alpha=args.alpha,
        norm_preserving=True,
        max_new_tokens=128,
    )
    print(steered_text)

    print("\nSmoke test finished.")


if __name__ == "__main__":
    main()
