#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hidden_state.modeling import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a direction-control steering config from a base config and run the pipeline."
        )
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="gsm8k_steering_exact")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--custom-tasks-dir", default="custom_task")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--mode",
        default="rotate_2d",
        choices=["none", "rotate_2d", "rotate_3d"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--polar-deg", type=float, default=None)
    parser.add_argument("--azimuth-deg", type=float, default=None)
    parser.add_argument(
        "--config-out",
        default=None,
        help="Optional path for the generated config JSON. Defaults to <output-dir>/direction_control_config.json",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Write the derived config and print the downstream run_pipeline command without executing it.",
    )
    return parser.parse_args()


def build_config(base_config_path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    payload["direction_control_mode"] = args.mode
    payload["direction_control_seed"] = int(args.seed)

    if args.angle_deg is None:
        payload.pop("direction_control_angle_deg", None)
    else:
        payload["direction_control_angle_deg"] = float(args.angle_deg)

    if args.polar_deg is None:
        payload.pop("direction_control_polar_deg", None)
    else:
        payload["direction_control_polar_deg"] = float(args.polar_deg)

    if args.azimuth_deg is None:
        payload.pop("direction_control_azimuth_deg", None)
    else:
        payload["direction_control_azimuth_deg"] = float(args.azimuth_deg)

    return payload


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_out = (
        Path(args.config_out).expanduser().resolve()
        if args.config_out is not None
        else output_dir / "direction_control_config.json"
    )
    config_out.parent.mkdir(parents=True, exist_ok=True)

    payload = build_config(base_config_path, args)
    config_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote derived config: {config_out}", flush=True)

    command = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "run_pipeline.py").resolve()),
        "--model-name",
        args.model_name,
        "--tasks",
        args.tasks,
        "--custom-tasks-dir",
        args.custom_tasks_dir,
        "--steering-config",
        str(config_out),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])

    print("Run command:", " ".join(command), flush=True)
    if args.print_only:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
