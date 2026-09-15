"""TeleLogs root-cause taxonomy and task-level scoring helpers.

The eight causes follow the public TeleLogs dataset card.  Dataset samples are
kept outside the repository because the official benchmark is gated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

ROOT_CAUSES: dict[str, dict[str, Any]] = {
    "C1": {
        "name": "excessive_vehicle_speed",
        "description": "Test vehicle speed exceeds 40 km/h.",
        "aliases": ("test vehicle speed exceeds 40", "speed exceeds 40"),
        "evidence_keywords": ("speed", "40 km/h", "vehicle"),
        "repair_keywords": ("reduce speed", "below 40", "speed cap", "slow down"),
    },
    "C2": {
        "name": "excessive_serving_cell_downtilt",
        "description": "Serving-cell downtilt is too large and weakens far-end coverage.",
        "aliases": ("downtilt angle is too large", "excessive downtilt"),
        "evidence_keywords": ("downtilt", "tilt", "far end", "weak coverage"),
        "repair_keywords": ("reduce downtilt", "retune tilt", "antenna alignment", "optimize downtilt"),
    },
    "C3": {
        "name": "excessive_serving_distance",
        "description": "Serving-cell coverage distance exceeds 1 km and yields poor RSRP.",
        "aliases": ("coverage distance exceeds 1", "over-shooting", "overshooting"),
        "evidence_keywords": ("distance", "1 km", "rsrp", "location"),
        "repair_keywords": ("nearer cell", "coverage", "handover", "site planning", "power"),
    },
    "C4": {
        "name": "non_colocated_cochannel_interference",
        "description": "A non-colocated co-frequency neighbor causes severe interference.",
        "aliases": ("non-colocated co-frequency", "overlapping coverage"),
        "evidence_keywords": ("co-frequency", "cochannel", "sinr", "interference", "neighbor"),
        "repair_keywords": ("frequency planning", "interference coordination", "change frequency", "icic"),
    },
    "C5": {
        "name": "pci_mod30_collision",
        "description": "Serving and neighbor PCI values collide modulo 30.",
        "aliases": ("same pci mod 30", "pci mod 30", "modulo 30"),
        "evidence_keywords": ("pci", "mod 30", "mod30", "reference signal"),
        "repair_keywords": ("pci planning", "change pci", "reassign pci", "avoid mod 30"),
    },
    "C6": {
        "name": "frequent_handovers",
        "description": "Frequent handovers degrade user performance.",
        "aliases": ("frequent handovers", "ping-pong handover"),
        "evidence_keywords": ("frequent handover", "handover count", "ping-pong", "ho event"),
        "repair_keywords": ("reduce handover", "hysteresis", "time-to-trigger", "ttt", "ping-pong"),
    },
    "C7": {
        "name": "misconfigured_handover_thresholds",
        "description": "Misconfigured handover thresholds degrade user performance.",
        "aliases": (
            "misconfigured handover threshold",
            "incorrect handover threshold",
            "neighboring cell provides higher throughput",
            "neighbor cell provides higher throughput",
        ),
        "evidence_keywords": ("handover threshold", "a3", "hysteresis", "time-to-trigger", "ttt"),
        "repair_keywords": ("tune threshold", "retune a3", "adjust hysteresis", "adjust ttt"),
    },
    "C8": {
        "name": "insufficient_scheduled_rbs",
        "description": "Average scheduled resource blocks are below 160.",
        "aliases": ("scheduled rbs are below 160", "scheduled resource blocks are below 160"),
        "evidence_keywords": ("resource block", "scheduled rb", "rbs", "prb", "160"),
        "repair_keywords": ("increase rb", "resource allocation", "scheduler", "at least 160", "capacity"),
    },
}

CAUSE_IDS = tuple(ROOT_CAUSES)


def _flatten_labels(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _flatten_labels(parsed)
        explicit = re.findall(r"(?i)\bC\s*([1-8])\b", text)
        if explicit:
            return [f"C{item}" for item in explicit]
        pieces = [piece.strip() for piece in re.split(r"[,;|\n]+", text) if piece.strip()]
        return pieces if len(pieces) > 1 else [text]
    if isinstance(value, dict):
        for key in ("root_causes", "root_cause", "labels", "label", "answer", "target"):
            if key in value:
                return _flatten_labels(value[key])
        return list(value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_labels(item))
        return flattened
    return [value]


def normalize_root_causes(value: Any, *, choices: list[str] | None = None) -> list[str]:
    """Normalize dataset or model labels to sorted ``C1`` ... ``C8`` IDs.

    Integer labels are treated as zero-based when choices are present, which
    matches common multiple-choice dataset exports. Otherwise 1..8 are treated
    as the public TeleLogs cause IDs.
    """

    normalized: set[str] = set()
    for raw in _flatten_labels(value):
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            if choices is not None and 0 <= raw < len(choices):
                normalized.update(normalize_root_causes(choices[raw]))
                continue
            number = raw
            if 1 <= number <= 8:
                normalized.add(f"C{number}")
            continue

        text = str(raw).strip()
        explicit = re.fullmatch(r"(?i)C?\s*([1-8])", text)
        if explicit:
            normalized.add(f"C{explicit.group(1)}")
            continue
        letter = re.fullmatch(r"(?i)[A-H]", text)
        if letter:
            normalized.add(f"C{ord(letter.group(0).upper()) - ord('A') + 1}")
            continue

        lowered = text.casefold()
        if choices is not None:
            for index, choice in enumerate(choices[:8]):
                if lowered == str(choice).strip().casefold():
                    normalized.add(f"C{index + 1}")
        for cause_id, spec in ROOT_CAUSES.items():
            name = str(spec["name"]).replace("_", " ")
            description = str(spec["description"])
            aliases = tuple(str(alias).casefold() for alias in spec.get("aliases", ()))
            if (
                lowered == name.casefold()
                or lowered in description.casefold()
                or description.casefold() in lowered
                or any(alias in lowered for alias in aliases)
            ):
                normalized.add(cause_id)
                continue
            keyword_hits = sum(keyword.casefold() in lowered for keyword in spec["evidence_keywords"])
            if keyword_hits >= 2:
                normalized.add(cause_id)

    return sorted(normalized, key=lambda item: int(item[1:]))


def set_f1(predicted: Iterable[str], expected: Iterable[str]) -> float:
    predicted_set = set(normalize_root_causes(list(predicted)))
    expected_set = set(normalize_root_causes(list(expected)))
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    true_positive = len(predicted_set & expected_set)
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(expected_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def recommended_repairs(cause_ids: Iterable[str]) -> list[str]:
    recommendations = {
        "C1": "Keep the test vehicle at or below 40 km/h and re-run the drive test.",
        "C2": "Reduce and re-validate the serving-cell downtilt using coverage measurements.",
        "C3": "Move service to a nearer cell or optimize coverage and mobility boundaries.",
        "C4": "Apply co-channel interference coordination or revise the frequency plan.",
        "C5": "Reassign PCI values so the serving and neighbor cells do not collide modulo 30.",
        "C6": "Reduce ping-pong handovers by tuning hysteresis and time-to-trigger.",
        "C7": "Correct the handover event thresholds and validate them with replayed traces.",
        "C8": "Increase scheduled RB allocation to at least 160 on average or relieve scheduler congestion.",
    }
    return [recommendations[cause_id] for cause_id in normalize_root_causes(list(cause_ids))]
