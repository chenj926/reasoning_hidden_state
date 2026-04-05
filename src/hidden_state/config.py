from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SteeringRuntimeConfig:
    precision: str = "bf16"
    steering_method: str = "none"  # none | vanilla | tgs
    alpha: float = 0.05
    k: int = 3
    norm_preserving: bool = True
    tgs_vector_path: str = ""
    use_chat_template: bool = False
    enable_thinking: Optional[bool] = None
    system_prompt: str = "You are a careful mathematical reasoning assistant."
    default_max_new_tokens: int = 256
    temperature_for_sampling: float = 0.7
    top_p_for_sampling: float = 0.95


def load_runtime_config() -> SteeringRuntimeConfig:
    cfg_path = os.environ.get("STEERING_CONFIG_JSON", "")
    if not cfg_path:
        return SteeringRuntimeConfig()
    payload = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    return SteeringRuntimeConfig(**payload)
