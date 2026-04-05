from __future__ import annotations

import math
import re
from typing import Optional

import sympy as sp

try:
    from latex2sympy2_extended import latex2sympy
except Exception:  # pragma: no cover
    latex2sympy = None

try:
    from lighteval.metrics.normalizations import math_normalizer as lighteval_math_normalizer
except Exception:  # pragma: no cover
    lighteval_math_normalizer = None


ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
LATEX_TEXT_RE = re.compile(r"^\\text\{(.+)\}$")


def _strip_dollars(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        return text[1:-1].strip()
    return text


def _strip_outer_text(text: str) -> str:
    m = LATEX_TEXT_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def _remove_common_latex_noise(text: str) -> str:
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\:": "",
        "\\%": "%",
        "\\$": "$",
        "\\displaystyle": "",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.strip()


def extract_answer_line(text: str) -> Optional[str]:
    matches: list[str] = []
    for line in text.splitlines():
        m = ANSWER_LINE_RE.search(line.strip())
        if m:
            matches.append(m.group(1).strip())
    if matches:
        return matches[-1]
    return None


def extract_last_boxed(text: str) -> Optional[str]:
    for token in ("\\boxed{", "\\fbox{"):
        start = text.rfind(token)
        if start == -1:
            continue
        idx = start + len(token)
        depth = 1
        chars = []
        while idx < len(text):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
            chars.append(ch)
            idx += 1
    return None


def extract_numeric_fallback(text: str) -> Optional[str]:
    matches = NUMERIC_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_final_answer(text: str) -> Optional[str]:
    answer = extract_answer_line(text)
    if answer:
        return answer.strip()
    answer = extract_last_boxed(text)
    if answer:
        return answer.strip()
    answer = extract_numeric_fallback(text)
    if answer:
        return answer.strip()
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty:
        return nonempty[-1]
    return None


def basic_normalize_math_answer(text: str) -> str:
    text = text.strip()
    text = _strip_dollars(text)
    text = _strip_outer_text(text)
    text = _remove_common_latex_noise(text)
    text = text.strip().rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


def normalize_math_answer(text: str) -> str:
    candidate = basic_normalize_math_answer(text)
    if lighteval_math_normalizer is not None:
        try:
            candidate = lighteval_math_normalizer(candidate)
        except Exception:
            pass
    return basic_normalize_math_answer(candidate)


def _to_sympy_expr(text: str):
    candidate = normalize_math_answer(text)
    if candidate == "":
        raise ValueError("Empty answer.")
    if latex2sympy is not None:
        try:
            return latex2sympy(candidate)
        except Exception:
            pass
    return sp.sympify(candidate)


def answers_equivalent(pred: Optional[str], gold: Optional[str], atol: float = 1e-8) -> bool:
    if pred is None or gold is None:
        return False

    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)
    if pred_norm == gold_norm:
        return True

    try:
        pred_expr = _to_sympy_expr(pred_norm)
        gold_expr = _to_sympy_expr(gold_norm)
        if sp.simplify(pred_expr - gold_expr) == 0:
            return True
    except Exception:
        pass

    try:
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        if math.isclose(pred_val, gold_val, rel_tol=0.0, abs_tol=atol):
            return True
    except Exception:
        pass

    return False
