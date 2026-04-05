from __future__ import annotations

from typing import Optional

import torch

from .modeling import LoadedModelBundle
from .steering_core import SteeringBundle, SteeringHookManager


def truncate_at_stops(text: str, stop_sequences: list[str] | None) -> str:
    if not stop_sequences:
        return text
    positions = [text.find(stop) for stop in stop_sequences if stop in text]
    if not positions:
        return text
    return text[: min(positions)]


@torch.inference_mode()
def generate_samples(
    bundle: LoadedModelBundle,
    prompt_text: str,
    *,
    steering_bundle: Optional[SteeringBundle],
    alpha: float,
    norm_preserving: bool,
    max_new_tokens: int,
    stop_sequences: list[str] | None,
    num_samples: int,
    temperature: float,
    top_p: float,
) -> tuple[list[str], list[list[int]], list[int]]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
    prompt_len = int(encoded['input_ids'].shape[1])

    context = (
        SteeringHookManager(model, steering_bundle, alpha=alpha, norm_preserving=norm_preserving)
        if steering_bundle is not None
        else torch.no_grad()
    )
    with context:
        outputs = model.generate(
            **encoded,
            do_sample=num_samples > 1,
            temperature=temperature if num_samples > 1 else None,
            top_p=top_p if num_samples > 1 else None,
            top_k=None if num_samples > 1 else 50,
            num_beams=1,
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    texts = []
    output_token_lists = []
    input_token_list = encoded['input_ids'][0].detach().cpu().tolist()
    for row in outputs:
        output_ids = row[prompt_len:].detach().cpu().tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        text = truncate_at_stops(text, stop_sequences)
        texts.append(text)
        output_token_lists.append(output_ids)
    return texts, output_token_lists, input_token_list
