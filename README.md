# qwen_steering_project

Minimal, local, text-only adaptation of hidden-state steering for `Qwen/Qwen3-0.6B`.

The default model ID in the runner scripts is now `Qwen/Qwen3-0.6B`.
If you previously built TGS artifacts for `Qwen/Qwen2-0.5B-Instruct`, rebuild them before running TGS with Qwen3:
the old vectors were extracted from a 24-layer / 896-d hidden space, while `Qwen3-0.6B` uses 28 layers / 1024-d.

This code is designed for:
- WSL2 / Linux
- single-GPU local inference
- hidden-state extraction + hidden-state injection during generation
- GSM8K and MATH-style evaluation with a Lighteval-compatible prompt/metric bridge

The benchmark runners in `scripts/` intentionally prioritize:
1. methodological transparency,
2. debuggability,
3. exact control of the steered generation path.

They do **not** implement the full `lighteval custom` model interface. Instead, they align
their prompts and dataset choices with the official Lighteval task definitions where possible,
and use a custom single-sample greedy protocol for MATH.

## Direction-Control Ablations

You can now rotate steering vectors at runtime without rebuilding the bundle. Add the following
optional fields to a steering config JSON:

- `direction_control_mode`: `"none"`, `"rotate_2d"`, or `"rotate_3d"`
- `direction_control_seed`: integer seed used for deterministic random angles and bases
- `direction_control_angle_deg`: 2D rotation angle in degrees; if omitted, one is sampled from `[0, 360)`
- `direction_control_polar_deg`: 3D polar angle away from the original steering direction
- `direction_control_azimuth_deg`: 3D azimuth angle around the original steering direction

`rotate_2d` preserves the original vector norm and rotates inside a plane spanned by the steering
vector and one orthogonal random direction. `rotate_3d` does the same inside a 3D subspace using
two orthogonal random directions. If the angle fields are omitted, the sampled angles are fully
determined by `direction_control_seed`, so the ablations stay reproducible.
