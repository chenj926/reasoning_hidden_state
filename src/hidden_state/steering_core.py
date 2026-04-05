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


def validate_steering_bundle_for_model(bundle: SteeringBundle, model, model_id: str) -> None:
    config = model.config
    expected_hidden_size = getattr(config, "hidden_size", None)
    expected_num_hidden_layers = getattr(config, "num_hidden_layers", None)

    bundle_model_id = bundle.metadata.get("model_id")
    bundle_hidden_size = bundle.metadata.get("hidden_size")
    bundle_num_hidden_layers = bundle.metadata.get("num_hidden_layers")

    if bundle_model_id and bundle_model_id != model_id:
        raise ValueError(
            "Steering bundle/model mismatch: "
            f"bundle was built for {bundle_model_id}, but runtime model is {model_id}. "
            "Rebuild the bundle for the target model."
        )

    if (
        bundle_hidden_size is not None
        and expected_hidden_size is not None
        and int(bundle_hidden_size) != int(expected_hidden_size)
    ):
        raise ValueError(
            "Steering bundle hidden size mismatch: "
            f"bundle metadata says {bundle_hidden_size}, but {model_id} expects {expected_hidden_size}. "
            "Rebuild the bundle for the target model."
        )

    if (
        bundle_num_hidden_layers is not None
        and expected_num_hidden_layers is not None
        and int(bundle_num_hidden_layers) != int(expected_num_hidden_layers)
    ):
        raise ValueError(
            "Steering bundle layer-count mismatch: "
            f"bundle metadata says {bundle_num_hidden_layers}, but {model_id} expects {expected_num_hidden_layers}. "
            "Rebuild the bundle for the target model."
        )

    if expected_num_hidden_layers is not None:
        invalid_layers = [
            layer_idx for layer_idx in bundle.layer_indices if layer_idx >= int(expected_num_hidden_layers)
        ]
        if invalid_layers:
            raise ValueError(
                "Steering bundle targets unavailable layers: "
                f"{invalid_layers[:5]} for model {model_id} with {expected_num_hidden_layers} layers."
            )

    if expected_hidden_size is not None:
        mismatched = [
            (layer_idx, int(vector.numel()))
            for layer_idx, vector in bundle.layer_vectors.items()
            if int(vector.numel()) != int(expected_hidden_size)
        ]
        if mismatched:
            layer_idx, actual_size = mismatched[0]
            raise ValueError(
                "Steering bundle vector size mismatch: "
                f"layer {layer_idx} has dim {actual_size}, but {model_id} expects {expected_hidden_size}. "
                "Rebuild the bundle for the target model."
            )


def _get_decoder_layers(model):
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "layers"):
        return base_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        "Unsupported model architecture for steering hooks: expected `.model.layers` or `.layers`."
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
        layers = _get_decoder_layers(self.model)
        for layer_idx in self.bundle.layer_indices:
            handle = layers[layer_idx].register_forward_hook(self._make_hook(self.bundle.layer_vectors[layer_idx]))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
