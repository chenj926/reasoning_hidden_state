from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .steering_core import SteeringBundle


@dataclass(frozen=True)
class DirectionControlSpec:
    mode: str = "none"  # none | rotate_2d | rotate_3d
    seed: int = 0
    angle_deg: float | None = None
    polar_deg: float | None = None
    azimuth_deg: float | None = None

    def is_active(self) -> bool:
        return self.mode != "none"


def apply_direction_control(bundle: SteeringBundle | None, spec: DirectionControlSpec) -> SteeringBundle | None:
    if bundle is None or not spec.is_active():
        return bundle

    if spec.mode == "rotate_2d":
        angle_deg = _resolve_2d_angle_deg(spec)
        rotated_vectors = {
            layer_idx: _rotate_vector_2d(vector, layer_idx=layer_idx, angle_deg=angle_deg, seed=spec.seed)
            for layer_idx, vector in bundle.layer_vectors.items()
        }
        metadata = {
            **bundle.metadata,
            "direction_control": {
                "mode": spec.mode,
                "seed": spec.seed,
                "angle_deg": angle_deg,
            },
        }
        return SteeringBundle(layer_vectors=rotated_vectors, source=bundle.source, metadata=metadata)

    if spec.mode == "rotate_3d":
        polar_deg, azimuth_deg = _resolve_3d_angles_deg(spec)
        rotated_vectors = {
            layer_idx: _rotate_vector_3d(
                vector,
                layer_idx=layer_idx,
                polar_deg=polar_deg,
                azimuth_deg=azimuth_deg,
                seed=spec.seed,
            )
            for layer_idx, vector in bundle.layer_vectors.items()
        }
        metadata = {
            **bundle.metadata,
            "direction_control": {
                "mode": spec.mode,
                "seed": spec.seed,
                "polar_deg": polar_deg,
                "azimuth_deg": azimuth_deg,
            },
        }
        return SteeringBundle(layer_vectors=rotated_vectors, source=bundle.source, metadata=metadata)

    raise ValueError(f"Unsupported direction control mode: {spec.mode}")


def describe_direction_control(spec: DirectionControlSpec) -> str:
    if spec.mode == "none":
        return "none"
    if spec.mode == "rotate_2d":
        angle_deg = _resolve_2d_angle_deg(spec)
        return f"rotate_2d(angle_deg={angle_deg:.3f}, seed={spec.seed})"
    if spec.mode == "rotate_3d":
        polar_deg, azimuth_deg = _resolve_3d_angles_deg(spec)
        return (
            "rotate_3d("
            f"polar_deg={polar_deg:.3f}, azimuth_deg={azimuth_deg:.3f}, seed={spec.seed}"
            ")"
        )
    return spec.mode


def _resolve_2d_angle_deg(spec: DirectionControlSpec) -> float:
    if spec.angle_deg is not None:
        return float(spec.angle_deg)
    return _sample_uniform_degrees(spec.seed, "rotate_2d_angle")


def _resolve_3d_angles_deg(spec: DirectionControlSpec) -> tuple[float, float]:
    if spec.polar_deg is None:
        polar_deg = _sample_uniform_polar_degrees(spec.seed, "rotate_3d_polar")
    else:
        polar_deg = float(spec.polar_deg)

    if spec.azimuth_deg is None:
        azimuth_deg = _sample_uniform_degrees(spec.seed, "rotate_3d_azimuth")
    else:
        azimuth_deg = float(spec.azimuth_deg)
    return polar_deg, azimuth_deg


def _rotate_vector_2d(vector: torch.Tensor, *, layer_idx: int, angle_deg: float, seed: int) -> torch.Tensor:
    base = vector.detach().cpu().float().flatten()
    base_norm = float(base.norm().item())
    if base_norm == 0.0:
        return vector.detach().cpu().clone()

    u0 = base / base_norm
    u1 = _sample_orthogonal_unit(
        refs=[u0],
        seed_parts=(seed, "rotate_2d_basis", layer_idx),
    )
    angle_rad = math.radians(angle_deg)
    rotated = math.cos(angle_rad) * u0 + math.sin(angle_rad) * u1
    rotated = rotated * base_norm
    return rotated.to(dtype=vector.dtype).view_as(vector).cpu()


def _rotate_vector_3d(
    vector: torch.Tensor,
    *,
    layer_idx: int,
    polar_deg: float,
    azimuth_deg: float,
    seed: int,
) -> torch.Tensor:
    base = vector.detach().cpu().float().flatten()
    base_norm = float(base.norm().item())
    if base_norm == 0.0:
        return vector.detach().cpu().clone()

    u0 = base / base_norm
    u1 = _sample_orthogonal_unit(
        refs=[u0],
        seed_parts=(seed, "rotate_3d_basis_a", layer_idx),
    )
    u2 = _sample_orthogonal_unit(
        refs=[u0, u1],
        seed_parts=(seed, "rotate_3d_basis_b", layer_idx),
    )

    polar_rad = math.radians(polar_deg)
    azimuth_rad = math.radians(azimuth_deg)
    tangent = math.cos(azimuth_rad) * u1 + math.sin(azimuth_rad) * u2
    rotated = math.cos(polar_rad) * u0 + math.sin(polar_rad) * tangent
    rotated = rotated * base_norm
    return rotated.to(dtype=vector.dtype).view_as(vector).cpu()


def _sample_orthogonal_unit(*, refs: list[torch.Tensor], seed_parts: tuple[object, ...], eps: float = 1e-8) -> torch.Tensor:
    shape = refs[0].shape
    normalized_refs = [ref / ref.norm().clamp_min(eps) for ref in refs]
    for attempt in range(32):
        candidate = torch.randn(shape, generator=_make_generator(*seed_parts, attempt), dtype=torch.float32)
        for ref in normalized_refs:
            candidate = candidate - torch.dot(candidate, ref) * ref
        norm = float(candidate.norm().item())
        if norm > eps:
            return candidate / norm
    raise RuntimeError("Failed to sample a stable orthogonal steering direction.")


def _sample_uniform_degrees(seed: int, label: str) -> float:
    value = torch.rand((), generator=_make_generator(seed, label), dtype=torch.float32).item()
    return float(value) * 360.0


def _sample_uniform_polar_degrees(seed: int, label: str) -> float:
    value = torch.rand((), generator=_make_generator(seed, label), dtype=torch.float32).item()
    cosine = 2.0 * float(value) - 1.0
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _make_generator(*parts: object) -> torch.Generator:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator
