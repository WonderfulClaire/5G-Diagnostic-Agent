"""Audit saved GRPO metrics for reward/correctness alignment and unsafe updates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from recipes.telelogs.training.objective import learning_route

UPDATABLE_ROUTES = {"rl_ready", "efficiency_rl"}


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    rewards = record.get("rewards")
    correctness = record.get("correctness_scores")
    if rewards is None or correctness is None:
        raise ValueError("metrics record must include rewards and correctness_scores")

    costs = record.get("efficiency_costs")
    recomputed = learning_route(
        rewards,
        correctness,
        efficiency_costs=costs,
    )
    stored = record.get("learning_route")
    updated = bool(record.get("optimizer_updated", False))

    issues: list[str] = []
    if stored is not None and stored != recomputed:
        issues.append("route_mismatch")
    if updated and recomputed not in UPDATABLE_ROUTES:
        issues.append("unsafe_optimizer_update")
    if not updated and recomputed in UPDATABLE_ROUTES:
        issues.append("missed_optimizer_update")

    return {
        "step": record.get("step"),
        "case_id": record.get("case_id"),
        "stored_route": stored,
        "recomputed_route": recomputed,
        "optimizer_updated": updated,
        "issues": issues,
    }


def audit_file(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                result = audit_record(record)
            except Exception as exc:
                result = {
                    "step": None,
                    "case_id": None,
                    "stored_route": None,
                    "recomputed_route": "invalid_record",
                    "optimizer_updated": False,
                    "issues": [f"parse_or_validation_error:{type(exc).__name__}"],
                }
            result["line"] = lineno
            rows.append(result)

    route_counts = Counter(row["recomputed_route"] for row in rows)
    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    return {
        "file": str(path),
        "records": len(rows),
        "route_counts": dict(sorted(route_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": [row for row in rows if row["issues"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute learning routes from saved GRPO metrics and audit unsafe updates."
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any audit issue is found",
    )
    args = parser.parse_args()

    report = audit_file(args.metrics)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report["issue_counts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
