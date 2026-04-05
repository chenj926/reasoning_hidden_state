from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PrecisionMode = Literal["fp16", "bf16", "8bit", "4bit"]


@dataclass
class LoadedModelBundle:
    model_id: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    precision: PrecisionMode


def _get_first_parameter_device(model: AutoModelForCausalLM) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _resolve_dtype(precision: PrecisionMode) -> Optional[torch.dtype]:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("bf16 was requested, but CUDA is not available.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 was requested, but torch.cuda.is_bf16_supported() returned False."
            )
        return torch.bfloat16
    if precision in {"8bit", "4bit"}:
        return None
    raise ValueError(f"Unknown precision mode: {precision}")


def load_model_and_tokenizer(
    model_id: str,
    precision: PrecisionMode = "bf16",
    device_map: str = "auto",
) -> LoadedModelBundle:
    model_kwargs: dict = {"device_map": device_map}

    if precision in {"8bit", "4bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "8bit/4bit precision requested but BitsAndBytesConfig is unavailable. "
                "Install bitsandbytes first."
            ) from exc

        if precision == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(precision)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    device = _get_first_parameter_device(model)

    if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return LoadedModelBundle(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        precision=precision,
    )
