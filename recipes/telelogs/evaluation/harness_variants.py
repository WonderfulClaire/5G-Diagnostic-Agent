"""Tool-schema variants for held-out harness generalization evaluation.

The environment itself stays canonical. Only the model-facing query tool names change.
Before execution, aliased tool calls are translated back to canonical names.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any


HARNESS_VARIANTS: dict[str, dict[str, str]] = {
    "canonical": {},
    "compact": {
        "query_radio_kpi": "radio",
        "query_cell_relation": "cells",
        "query_mobility": "mobility",
        "query_antenna": "antenna",
        "query_handover": "handover",
        "query_resource": "resource",
    },
    "alternate": {
        "query_radio_kpi": "inspect_radio_metrics",
        "query_cell_relation": "inspect_neighbor_cells",
        "query_mobility": "inspect_mobility_state",
        "query_antenna": "inspect_antenna_config",
        "query_handover": "inspect_handover_events",
        "query_resource": "inspect_resource_allocation",
    },
}

_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def available_variants() -> tuple[str, ...]:
    return tuple(HARNESS_VARIANTS)


def alias_tool_schemas(
    schemas: list[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    """Return model-facing schemas with query tools renamed.

    submit_diagnosis deliberately remains stable so this ablation changes the
    evidence-query interface without also changing the final output contract.
    """
    if variant not in HARNESS_VARIANTS:
        raise ValueError(f"unknown harness variant: {variant}")
    mapping = HARNESS_VARIANTS[variant]
    aliased = copy.deepcopy(schemas)
    for schema in aliased:
        function = schema.get("function", {})
        name = function.get("name")
        if name in mapping:
            function["name"] = mapping[name]
    return aliased


def canonicalize_tool_call_text(text: str, variant: str) -> str:
    """Translate model-facing aliases back to the environment's canonical names."""
    if variant not in HARNESS_VARIANTS:
        raise ValueError(f"unknown harness variant: {variant}")
    reverse = {alias: canonical for canonical, alias in HARNESS_VARIANTS[variant].items()}
    if not reverse:
        return text

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return match.group(0)
        name = payload.get("name")
        if name not in reverse:
            return match.group(0)
        payload["name"] = reverse[name]
        return "<tool_call>" + json.dumps(payload, ensure_ascii=False) + "</tool_call>"

    return _BLOCK_RE.sub(replace, text)
