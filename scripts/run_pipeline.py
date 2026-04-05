#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CUSTOM_MODEL_PATH = PROJECT_ROOT / 'custom_model' / 'steered_qwen_model.py'
DEFAULT_CUSTOM_TASKS_MODULE = 'custom_task'

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

from src.hidden_state.modeling import DEFAULT_MODEL_NAME
from src.hidden_state.predictions_export import latest_saved_date_id, write_predictions_jsonl


def _resolve_custom_tasks_source(custom_tasks: str, tasks: str) -> str:
    candidate = Path(custom_tasks)
    if not candidate.exists():
        return custom_tasks

    if candidate.is_file():
        return str(candidate.resolve())

    task_names = [item.strip().split('|')[-1].split(':')[0] for item in tasks.split(',') if item.strip()]
    if len(task_names) == 1:
        task_file = candidate / f'{task_names[0]}.py'
        if task_file.exists():
            return str(task_file.resolve())

    init_file = candidate / '__init__.py'
    if init_file.exists():
        parent = str(candidate.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return str(init_file.resolve())

    raise ValueError(
        f'Custom tasks source {candidate} is a directory, but it is not an importable package '
        'and no matching single-task file was found. Pass a module name, a .py file, or a '
        'package directory with __init__.py.'
    )


def _normalize_tasks(tasks: str) -> str:
    normalized = []
    for item in tasks.split(','):
        task = item.strip()
        if not task:
            continue
        if '|' in task:
            normalized.append(task)
        else:
            normalized.append(f'custom|{task}|0')
    return ','.join(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--custom-tasks-dir', default=DEFAULT_CUSTOM_TASKS_MODULE)
    parser.add_argument('--steering-config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    steering_config_path = Path(args.steering_config).expanduser().resolve()
    if not steering_config_path.exists():
        raise FileNotFoundError(
            f'Steering config not found: {steering_config_path}. '
            'Create the JSON config first or pass an existing config path.'
        )

    normalized_tasks = _normalize_tasks(args.tasks)
    custom_tasks_source = _resolve_custom_tasks_source(args.custom_tasks_dir, normalized_tasks)

    os.environ['STEERING_CONFIG_JSON'] = str(steering_config_path)

    evaluation_tracker = EvaluationTracker(output_dir=args.output_dir, save_details=True)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.CUSTOM,
        custom_tasks_directory=custom_tasks_source,
        max_samples=args.max_samples,
    )
    model_config = CustomModelConfig(
        model_name=args.model_name,
        model_definition_file_path=str(CUSTOM_MODEL_PATH),
    )
    pipeline = Pipeline(
        tasks=normalized_tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    pipeline.evaluate()
    pipeline.save_and_push_results()
    date_id = latest_saved_date_id(args.output_dir, args.model_name)
    if date_id is not None:
        written_files = write_predictions_jsonl(
            evaluation_tracker.details,
            output_dir=args.output_dir,
            model_name=args.model_name,
            date_id=date_id,
        )
        for path in written_files:
            print(f"Saved predictions JSONL: {path}", flush=True)
    pipeline.show_results()


if __name__ == '__main__':
    main()
