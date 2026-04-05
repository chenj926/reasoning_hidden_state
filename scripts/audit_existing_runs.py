#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


def last_number(text: str | None):
    if text is None:
        return None
    matches = re.findall(r'-?\d+(?:\.\d+)?', str(text).replace(',', ''))
    return matches[-1] if matches else None


def analyze_run_dir(run_dir: Path) -> dict:
    preds_path = run_dir / 'predictions.jsonl'
    if not preds_path.exists():
        return {}
    total = 0
    recorded_correct = 0
    exact_match = 0
    numeric_lastnum_eq = 0
    answer_line_count = 0
    samples = []
    with preds_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            total += 1
            recorded_correct += int(bool(rec.get('is_correct', False)))
            gold = rec.get('gold_answer_extracted')
            pred = rec.get('prediction_extracted')
            if str(gold).strip() == str(pred).strip():
                exact_match += 1
            if last_number(gold) is not None and last_number(gold) == last_number(pred):
                numeric_lastnum_eq += 1
            if 'ANSWER:' in str(rec.get('prediction_raw', '')):
                answer_line_count += 1
            if idx < 3:
                samples.append({
                    'record_id': rec.get('record_id'),
                    'gold_answer_extracted': gold,
                    'prediction_extracted': pred,
                    'recorded_is_correct': rec.get('is_correct'),
                })
    return {
        'run_name': run_dir.name,
        'num_examples': total,
        'recorded_correct': recorded_correct,
        'recorded_accuracy': recorded_correct / total if total else 0.0,
        'string_exact_match': exact_match,
        'string_exact_match_rate': exact_match / total if total else 0.0,
        'numeric_lastnum_eq': numeric_lastnum_eq,
        'numeric_lastnum_eq_rate': numeric_lastnum_eq / total if total else 0.0,
        'answer_line_count': answer_line_count,
        'answer_line_rate': answer_line_count / total if total else 0.0,
        'samples': samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-zip', required=True)
    args = parser.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix='runs_audit_'))
    try:
        with zipfile.ZipFile(args.runs_zip) as zf:
            zf.extractall(tmpdir)
        summaries = []
        for subdir in sorted(p for p in tmpdir.iterdir() if p.is_dir()):
            summaries.append(analyze_run_dir(subdir))
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
