"""Helpers for turning rich reward diagnostics into scalar validation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


def _flatten_numeric_leaves(value: Any, prefix: str) -> dict[str, float]:
    """Return finite numeric leaves from a nested diagnostic value."""
    if isinstance(value, Mapping):
        flattened: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_leaves(child, child_prefix))
        return flattened

    if isinstance(value, Real) and math.isfinite(float(value)):
        return {prefix: float(value)}

    return {}


def build_numeric_validation_metrics(
    reward_extra_infos: Mapping[str, Sequence[Any]],
) -> dict[str, list[float]]:
    """Select scalar metrics that are numeric and present for every sample.

    Agent environments may attach rich dictionaries, strings, and lists to
    ``reward_extra_info`` for trajectory auditing. veRL's validation reducer
    applies NumPy means to every supplied field, so passing those objects
    raises at the first scheduled validation. Nested numeric score components
    are flattened and retained; non-numeric or partially missing fields remain
    available in trajectory dumps but are excluded from metric aggregation.
    """
    if not reward_extra_infos:
        return {}

    lengths = {len(values) for values in reward_extra_infos.values()}
    if len(lengths) != 1:
        raise ValueError(f"validation metric fields must be aligned, got lengths={sorted(lengths)}")

    sample_count = lengths.pop()
    if sample_count == 0:
        return {}

    sample_metrics: list[dict[str, float]] = []
    for sample_idx in range(sample_count):
        flattened: dict[str, float] = {}
        for key, values in reward_extra_infos.items():
            flattened.update(_flatten_numeric_leaves(values[sample_idx], str(key)))
        sample_metrics.append(flattened)

    common_names = set(sample_metrics[0])
    for metrics in sample_metrics[1:]:
        common_names.intersection_update(metrics)

    return {name: [metrics[name] for metrics in sample_metrics] for name in sorted(common_names)}
