from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


@dataclass
class LoadedModelBundle:
    model_id: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    precision: str

    def architecture_metadata(self) -> dict[str, Any]:
        config = self.model.config
        hidden_size = getattr(config, "hidden_size", None)
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", None)
        return {
            "model_id": self.model_id,
            "model_type": str(getattr(config, "model_type", "unknown")),
            "hidden_size": int(hidden_size) if hidden_size is not None else None,
            "num_hidden_layers": int(num_hidden_layers) if num_hidden_layers is not None else None,
            "max_position_embeddings": int(max_position_embeddings) if max_position_embeddings is not None else None,
        }


def _get_first_parameter_device(model) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _resolve_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("bf16 requested, but CUDA is unavailable.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested, but the current CUDA device does not support bf16.")
        return torch.bfloat16
    if precision in {"8bit", "4bit"}:
        return None
    raise ValueError(f"Unknown precision: {precision}")


def load_model_and_tokenizer(model_id: str, precision: str = "bf16", device_map: str = "auto") -> LoadedModelBundle:
    model_kwargs: dict = {"device_map": device_map}

    if precision in {"8bit", "4bit"}:
        from transformers import BitsAndBytesConfig

        if precision == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        model_kwargs["dtype"] = _resolve_dtype(precision)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return LoadedModelBundle(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=_get_first_parameter_device(model),
        precision=precision,
    )
