from __future__ import annotations

from typing import Iterable, Optional

import torch
from tqdm.auto import tqdm

from .modeling import LoadedModelBundle
from .prompting import build_prompt_pair_for_question, maybe_apply_chat_template
from .steering_core import SteeringBundle


@torch.inference_mode()
def extract_final_prompt_token_hidden_states(
    bundle: LoadedModelBundle,
    prompt_text: str,
    *,
    layer_indices: Optional[Iterable[int]] = None,
) -> dict[int, torch.Tensor]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
    final_prompt_index = int(encoded["input_ids"].shape[1] - 1)

    outputs = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = range(num_layers)

    collected: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        collected[layer_idx] = hidden_states[layer_idx + 1][0, final_prompt_index, :].detach().float().cpu()
    return collected


def difference_dict(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    if set(a.keys()) != set(b.keys()):
        raise ValueError("Layer sets do not match.")
    return {layer_idx: a[layer_idx] - b[layer_idx] for layer_idx in a.keys()}


def build_vanilla_bundle_for_question(
    bundle: LoadedModelBundle,
    question: str,
    *,
    benchmark_style: str,
    use_chat_template: bool,
    enable_thinking: bool | None,
    system_prompt: str,
) -> SteeringBundle:
    pair = build_prompt_pair_for_question(question, benchmark_style)
    cot_prompt = maybe_apply_chat_template(
        bundle.tokenizer,
        pair.cot_prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    norm_prompt = maybe_apply_chat_template(
        bundle.tokenizer,
        pair.norm_prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    cot_states = extract_final_prompt_token_hidden_states(bundle, cot_prompt)
    norm_states = extract_final_prompt_token_hidden_states(bundle, norm_prompt)
    return SteeringBundle(
        layer_vectors=difference_dict(cot_states, norm_states),
        source="vanilla",
        metadata={
            **bundle.architecture_metadata(),
            "question": question,
            "benchmark_style": benchmark_style,
            "use_chat_template": bool(use_chat_template),
            "enable_thinking": enable_thinking,
            "system_prompt": system_prompt,
        },
    )


def build_tgs_bundle(
    bundle: LoadedModelBundle,
    questions: list[str],
    *,
    benchmark_style: str,
    use_chat_template: bool,
    enable_thinking: bool | None,
    system_prompt: str,
) -> SteeringBundle:
    accumulators: dict[int, torch.Tensor] = {}
    count = 0
    for question in tqdm(questions, desc="Building TGS bundle"):
        pair = build_prompt_pair_for_question(question, benchmark_style)
        cot_prompt = maybe_apply_chat_template(
            bundle.tokenizer,
            pair.cot_prompt,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        norm_prompt = maybe_apply_chat_template(
            bundle.tokenizer,
            pair.norm_prompt,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        cot_states = extract_final_prompt_token_hidden_states(bundle, cot_prompt)
        norm_states = extract_final_prompt_token_hidden_states(bundle, norm_prompt)
        diff = difference_dict(cot_states, norm_states)
        if not accumulators:
            accumulators = {layer_idx: torch.zeros_like(vec) for layer_idx, vec in diff.items()}
        for layer_idx, vec in diff.items():
            accumulators[layer_idx] += vec
        count += 1

    if count == 0:
        raise ValueError("Cannot build a TGS bundle from an empty question list.")

    averaged = {layer_idx: vec / count for layer_idx, vec in accumulators.items()}
    return SteeringBundle(
        layer_vectors=averaged,
        source="tgs",
        metadata={
            **bundle.architecture_metadata(),
            "auxiliary_count": count,
            "benchmark_style": benchmark_style,
            "use_chat_template": bool(use_chat_template),
            "enable_thinking": enable_thinking,
            "system_prompt": system_prompt,
        },
    )
