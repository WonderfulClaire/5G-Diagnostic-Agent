#!/usr/bin/env python3
"""Summarize Agent-R1 TeleLogs validation trajectory dumps.

The trainer writes one JSON object per trajectory.  Environment diagnostics
are attached to individual steps, with the structured diagnosis nested in the
``tool_calls`` list of the terminal step.  This module converts those raw
traces into resume-auditable task and tool-use metrics without requiring the
gated source dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from recipes.telelogs.constants import ROOT_CAUSES, normalize_root_causes, set_f1


def _jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.jsonl"), key=lambda path: int(path.stem))


def load_entries(input_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for file_path in _jsonl_files(input_path):
        with file_path.open("r", encoding="utf-8") as handle:
            entries.extend(json.loads(line) for line in handle if line.strip())
    if not entries:
        raise ValueError(f"no validation entries found under {input_path}")
    return entries


def _iter_tool_infos(entry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for step in entry.get("steps", []):
        infos = step.get("tool_calls", [])
        if isinstance(infos, dict):
            infos = [infos]
        if isinstance(infos, list):
            yield from (info for info in infos if isinstance(info, dict))


def _terminal_submission(entry: dict[str, Any]) -> dict[str, Any] | None:
    submissions = [info for info in _iter_tool_infos(entry) if "predicted_root_causes" in info]
    return submissions[-1] if submissions else None


def trajectory_row(entry: dict[str, Any]) -> dict[str, Any]:
    expected = normalize_root_causes(entry.get("gts"))
    submission = _terminal_submission(entry)
    predicted = normalize_root_causes(submission.get("predicted_root_causes")) if submission else []
    tool_infos = list(_iter_tool_infos(entry))
    query_tools = [
        info.get("tool")
        for info in tool_infos
        if isinstance(info.get("tool"), str) and str(info["tool"]).startswith("query_")
    ]
    failures = sum(bool(info.get("error")) for info in tool_infos)
    missing_submission = submission is None
    if missing_submission:
        failures += 1
    components = submission.get("score_components", {}) if submission else {}
    return {
        "trajectory_uid": str(entry.get("trajectory_uid", "")),
        "step": int(entry.get("step", -1)),
        "ground_truth": expected,
        "predicted_root_causes": predicted,
        "exact": float(set(predicted) == set(expected)),
        "root_cause_f1": set_f1(predicted, expected),
        "submitted": float(not missing_submission),
        "num_steps": int(entry.get("num_steps", len(entry.get("steps", [])))),
        "query_count": len(query_tools),
        "tool_failure_count": failures,
        "reward": float(entry.get("score", 0.0)),
        "evidence_groundedness": float(components.get("evidence_groundedness", 0.0)),
        "cause_evidence_support": float(components.get("cause_evidence_support", 0.0)),
        "repair_relevance": float(components.get("repair_relevance", 0.0)),
        "tool_efficiency": float(components.get("tool_efficiency", 0.0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero trajectories")
    true_positive = false_positive = false_negative = 0
    per_cause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        predicted = set(row["predicted_root_causes"])
        expected = set(row["ground_truth"])
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        for cause in expected:
            per_cause[cause].append(row)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    micro_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    total_queries = sum(row["query_count"] for row in rows)
    summary = {
        "step": rows[0]["step"],
        "trajectories": len(rows),
        "exact_match": mean(row["exact"] for row in rows),
        "macro_f1": mean(row["root_cause_f1"] for row in rows),
        "micro_f1": micro_f1,
        "submission_rate": mean(row["submitted"] for row in rows),
        "avg_reward": mean(row["reward"] for row in rows),
        "avg_query_count": mean(row["query_count"] for row in rows),
        "avg_num_steps": mean(row["num_steps"] for row in rows),
        "tool_call_failure_rate": sum(row["tool_failure_count"] for row in rows) / max(1, total_queries),
        "evidence_groundedness": mean(row["evidence_groundedness"] for row in rows),
        "cause_evidence_support": mean(row["cause_evidence_support"] for row in rows),
        "repair_relevance": mean(row["repair_relevance"] for row in rows),
        "tool_efficiency": mean(row["tool_efficiency"] for row in rows),
        "per_cause": {},
    }
    for cause in ROOT_CAUSES:
        cause_rows = per_cause.get(cause, [])
        if cause_rows:
            summary["per_cause"][cause] = {
                "samples": len(cause_rows),
                "exact_match": mean(row["exact"] for row in cause_rows),
                "macro_f1": mean(row["root_cause_f1"] for row in cause_rows),
            }
    return summary


def summarize_step(entries: list[dict[str, Any]], step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [trajectory_row(entry) for entry in entries if int(entry.get("step", -1)) == step]
    return summarize_rows(rows), rows


def compare_summaries(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, float]:
    fields = (
        "exact_match",
        "macro_f1",
        "micro_f1",
        "submission_rate",
        "avg_reward",
        "avg_query_count",
        "tool_call_failure_rate",
        "evidence_groundedness",
        "cause_evidence_support",
        "repair_relevance",
        "tool_efficiency",
    )
    return {field: float(final[field]) - float(baseline[field]) for field in fields}


def markdown_report(baseline: dict[str, Any], final: dict[str, Any], delta: dict[str, float]) -> str:
    fields = (
        "trajectories",
        "exact_match",
        "macro_f1",
        "micro_f1",
        "submission_rate",
        "avg_reward",
        "avg_query_count",
        "avg_num_steps",
        "tool_call_failure_rate",
        "evidence_groundedness",
        "cause_evidence_support",
        "repair_relevance",
        "tool_efficiency",
    )
    lines = [
        "# TeleLogs validation comparison",
        "",
        "> Both columns come from the same configured validation split. "
        "Step 0 is the frozen base-model baseline; the final step is the trained policy.",
        "",
        "| Metric | Step 0 | Final | Delta |",
        "|---|---:|---:|---:|",
    ]
    for field in fields:
        before = baseline[field]
        after = final[field]
        if field == "trajectories":
            lines.append(f"| {field} | {before} | {after} | — |")
        else:
            lines.append(f"| {field} | {before:.4f} | {after:.4f} | {delta.get(field, 0.0):+.4f} |")
    lines.extend(
        [
            "",
            "## Per-cause result",
            "",
            "| Cause | Samples | Step 0 EM | Final EM | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cause in ROOT_CAUSES:
        before = baseline["per_cause"].get(cause, {})
        after = final["per_cause"].get(cause, {})
        if not before and not after:
            continue
        before_em = float(before.get("exact_match", 0.0))
        after_em = float(after.get("exact_match", 0.0))
        samples = after.get("samples", before.get("samples", 0))
        change = after_em - before_em
        lines.append(f"| {cause} | {samples} | {before_em:.4f} | {after_em:.4f} | {change:+.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TeleLogs Agent-R1 validation dumps")
    parser.add_argument("--input", type=Path, required=True, help="Validation JSONL directory or file")
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--final-step", type=int, default=None, help="Defaults to the largest available step")
    parser.add_argument("--output-dir", type=Path, default=Path("results/telelogs_validation"))
    args = parser.parse_args()

    entries = load_entries(args.input.expanduser().resolve())
    available_steps = sorted({int(entry.get("step", -1)) for entry in entries})
    final_step = args.final_step if args.final_step is not None else available_steps[-1]
    baseline, baseline_rows = summarize_step(entries, args.baseline_step)
    final, final_rows = summarize_step(entries, final_step)
    delta = compare_summaries(baseline, final)
    payload = {
        "source": str(args.input),
        "available_steps": available_steps,
        "baseline": baseline,
        "final": final,
        "delta": delta,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown_report(baseline, final, delta), encoding="utf-8")
    with (args.output_dir / "trajectory_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in baseline_rows + final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
