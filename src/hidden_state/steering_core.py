from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class SteeringBundle:
    layer_vectors: dict[int, torch.Tensor]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self.layer_vectors.keys())

    def vector_norms(self) -> dict[int, float]:
        return {idx: float(vec.float().norm().item()) for idx, vec in self.layer_vectors.items()}


def select_last_k_layers(bundle: SteeringBundle, k: int) -> SteeringBundle:
    available = sorted(bundle.layer_vectors.keys())
    if k <= 0 or k > len(available):
        raise ValueError(f"Invalid k={k} for bundle with {len(available)} layers.")
    selected = available[-k:]
    return SteeringBundle(
        layer_vectors={idx: bundle.layer_vectors[idx] for idx in selected},
        source=bundle.source,
        metadata={**bundle.metadata, "selected_last_k": k},
    )


def save_steering_bundle(bundle: SteeringBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_vectors": {str(k): v.detach().cpu() for k, v in bundle.layer_vectors.items()},
        "source": bundle.source,
        "metadata": bundle.metadata,
    }
    torch.save(payload, path)



def load_steering_bundle(path: str | Path) -> SteeringBundle:
    payload = torch.load(Path(path), map_location="cpu")
    return SteeringBundle(
        layer_vectors={int(k): v.detach().cpu() for k, v in payload["layer_vectors"].items()},
        source=payload.get("source", "unknown"),
        metadata=payload.get("metadata", {}),
    )


class SteeringHookManager(AbstractContextManager):
    def __init__(self, model, steering_bundle: SteeringBundle, alpha: float, norm_preserving: bool = True, eps: float = 1e-6):
        self.model = model
        self.bundle = steering_bundle
        self.alpha = float(alpha)
        self.norm_preserving = bool(norm_preserving)
        self.eps = float(eps)
        self.handles = []

    def _make_hook(self, vector_cpu: torch.Tensor):
        def hook(module, args, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Expected tensor output from decoder layer, got {type(output)}")
            vector = vector_cpu.to(device=output.device, dtype=output.dtype).view(1, 1, -1)
            steered = output + self.alpha * vector
            if self.norm_preserving:
                original_norm = output.norm(dim=-1, keepdim=True).clamp_min(self.eps)
                steered_norm = steered.norm(dim=-1, keepdim=True).clamp_min(self.eps)
                steered = steered * (original_norm / steered_norm)
            return steered
        return hook

    def __enter__(self):
        layers = self.model.model.layers
        for layer_idx in self.bundle.layer_indices:
            handle = layers[layer_idx].register_forward_hook(self._make_hook(self.bundle.layer_vectors[layer_idx]))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
