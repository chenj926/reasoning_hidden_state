from __future__ import annotations

from typing import Iterable, Optional

import torch

from .modeling import LoadedModelBundle
from .steering import SteeringBundle, SteeringHookManager


def _truncate_at_stop_sequences(text: str, stop_sequences: Optional[Iterable[str]]) -> str:
    if not stop_sequences:
        return text
    stop_positions = [text.find(stop) for stop in stop_sequences if stop in text]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


@torch.inference_mode()
def generate_text(
    bundle: LoadedModelBundle,
    prompt_text: str,
    *,
    steering_bundle: Optional[SteeringBundle] = None,
    alpha: float = 0.0,
    norm_preserving: bool = True,
    max_new_tokens: int = 256,
    stop_sequences: Optional[list[str]] = None,
) -> str:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])

    context = (
        SteeringHookManager(
            model=model,
            steering_bundle=steering_bundle,
            alpha=alpha,
            norm_preserving=norm_preserving,
        )
        if steering_bundle is not None
        else torch.no_grad()
    )

    with context:
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = generated[0, prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _truncate_at_stop_sequences(text, stop_sequences)


@torch.inference_mode()
def next_token_logits(
    bundle: LoadedModelBundle,
    prompt_text: str,
    *,
    steering_bundle: Optional[SteeringBundle] = None,
    alpha: float = 0.0,
    norm_preserving: bool = True,
) -> torch.Tensor:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    context = (
        SteeringHookManager(
            model=model,
            steering_bundle=steering_bundle,
            alpha=alpha,
            norm_preserving=norm_preserving,
        )
        if steering_bundle is not None
        else torch.no_grad()
    )

    with context:
        outputs = model(**encoded, use_cache=False, return_dict=True)

    return outputs.logits[0, -1, :].detach().float().cpu()
