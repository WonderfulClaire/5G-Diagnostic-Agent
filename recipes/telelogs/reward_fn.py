"""Fallback final-answer reward for direct TeleLogs generations."""

from __future__ import annotations

import json
import re
from typing import Any

from recipes.telelogs.constants import normalize_root_causes, set_f1


def _prediction_from_text(solution_str: str) -> list[str]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", solution_str, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend(re.findall(r"\{.*?\}", solution_str, flags=re.DOTALL))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        predicted = normalize_root_causes(payload)
        if predicted:
            return predicted
    return normalize_root_causes(solution_str)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    del data_source, extra_info, kwargs
    return float(set_f1(_prediction_from_text(solution_str), normalize_root_causes(ground_truth)))
