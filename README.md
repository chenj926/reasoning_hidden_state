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
