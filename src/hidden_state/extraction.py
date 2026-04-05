from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import torch
from tqdm.auto import tqdm

from .modeling import LoadedModelBundle
from .prompting import build_prompt_pair
from .steering import SteeringBundle


@torch.inference_mode()
def extract_final_prompt_token_hidden_states(
    bundle: LoadedModelBundle,
    prompt_text: str,
    layer_indices: Optional[Iterable[int]] = None,
) -> dict[int, torch.Tensor]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    final_prompt_index = int(encoded["input_ids"].shape[1] - 1)

    # hiddent state implementation
    outputs = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = range(num_layers)

    collected: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        # hidden_states[0] is the embedding output; hidden_states[layer_idx + 1] is decoder layer output.
        collected[layer_idx] = (
            hidden_states[layer_idx + 1][0, final_prompt_index, :]
            .detach()
            .float()
            .cpu()
        )
    return collected


def difference_dict(
    a: dict[int, torch.Tensor],
    b: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    if set(a) != set(b):
        raise ValueError("Layer sets do not match for difference computation.")
    return {layer_idx: a[layer_idx] - b[layer_idx] for layer_idx in a.keys()}


def build_vanilla_steering_bundle(
    bundle: LoadedModelBundle,
    question: str,
    *,
    use_chat_template: bool = True,
    system_prompt: str = "You are a careful mathematical reasoning assistant.",
) -> tuple[SteeringBundle, dict]:
    prompts = build_prompt_pair(
        bundle.tokenizer,
        question,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
    cot_states = extract_final_prompt_token_hidden_states(bundle, prompts.cot_model_prompt)
    norm_states = extract_final_prompt_token_hidden_states(bundle, prompts.norm_model_prompt)
    diff = difference_dict(cot_states, norm_states)
    steering_bundle = SteeringBundle(
        layer_vectors=diff,
        source="vanilla",
        metadata={
            "question": question,
            "use_chat_template": use_chat_template,
        },
    )
    return steering_bundle, {
        "cot_model_prompt": prompts.cot_model_prompt,
        "norm_model_prompt": prompts.norm_model_prompt,
        "cot_user_prompt": prompts.cot_user_prompt,
        "norm_user_prompt": prompts.norm_user_prompt,
    }


def build_tgs_steering_bundle(
    bundle: LoadedModelBundle,
    questions: list[str],
    *,
    use_chat_template: bool = True,
    system_prompt: str = "You are a careful mathematical reasoning assistant.",
    progress_desc: str = "Building TGS bundle",
) -> SteeringBundle:
    accumulators: dict[int, torch.Tensor] = {}
    count = 0

    for question in tqdm(questions, desc=progress_desc):
        pair = build_prompt_pair(
            bundle.tokenizer,
            question,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        cot_states = extract_final_prompt_token_hidden_states(bundle, pair.cot_model_prompt)
        norm_states = extract_final_prompt_token_hidden_states(bundle, pair.norm_model_prompt)
        diff = difference_dict(cot_states, norm_states)

        if not accumulators:
            accumulators = {layer_idx: torch.zeros_like(vec) for layer_idx, vec in diff.items()}

        for layer_idx, vec in diff.items():
            accumulators[layer_idx] += vec
        count += 1

    if count == 0:
        raise ValueError("TGS auxiliary question list is empty.")

    averaged = {layer_idx: vec / count for layer_idx, vec in accumulators.items()}
    return SteeringBundle(
        layer_vectors=averaged,
        source="tgs",
        metadata={
            "auxiliary_count": count,
            "use_chat_template": use_chat_template,
        },
    )
