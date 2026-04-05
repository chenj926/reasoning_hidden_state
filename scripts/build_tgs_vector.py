#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hidden_state.extraction import build_tgs_bundle
from src.hidden_state.modeling import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from src.hidden_state.steering_core import save_steering_bundle


def load_questions(aux_source: str, count: int, offset: int) -> list[str]:
    if aux_source == 'math_train':
        subjects = ['algebra','geometry','number_theory','prealgebra','precalculus','counting_and_probability','intermediate_algebra']
        questions = []
        for subject in subjects:
            ds = load_dataset('DigitalLearningGmbH/MATH-lighteval', subject, split='train')
            for row in ds:
                questions.append(row['problem'])
        return questions[offset:offset+count]
    if aux_source == 'gsm8k_train':
        ds = load_dataset('openai/gsm8k', 'main', split='train')
        questions = [row['question'] for row in ds]
        return questions[offset:offset+count]
    raise ValueError(f'Unknown aux_source: {aux_source}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--precision', default='bf16', choices=['fp16','bf16','8bit','4bit'])
    parser.add_argument('--aux-source', default='math_train', choices=['math_train','gsm8k_train'])
    parser.add_argument('--aux-count', type=int, default=32)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--benchmark-style', default='math_greedy', choices=['gsm8k','math_stock','math_greedy'])
    parser.add_argument('--use-chat-template', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--system-prompt', default='You are a careful mathematical reasoning assistant.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    questions = load_questions(args.aux_source, args.aux_count, args.offset)
    bundle = load_model_and_tokenizer(args.model, precision=args.precision)
    model_metadata = bundle.architecture_metadata()
    tgs = build_tgs_bundle(
        bundle,
        questions,
        benchmark_style=args.benchmark_style,
        use_chat_template=args.use_chat_template,
        enable_thinking=args.enable_thinking,
        system_prompt=args.system_prompt,
    )
    save_steering_bundle(tgs, args.output)
    meta = {
        'model': args.model,
        'model_type': model_metadata['model_type'],
        'hidden_size': model_metadata['hidden_size'],
        'num_hidden_layers': model_metadata['num_hidden_layers'],
        'precision': args.precision,
        'aux_source': args.aux_source,
        'aux_count': args.aux_count,
        'offset': args.offset,
        'benchmark_style': args.benchmark_style,
        'use_chat_template': args.use_chat_template,
        'enable_thinking': args.enable_thinking,
        'vector_norms': tgs.vector_norms(),
        'layer_indices': tgs.layer_indices,
        'metadata': tgs.metadata,
    }
    Path(args.output).with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
