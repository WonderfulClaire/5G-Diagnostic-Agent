#!/usr/bin/env python3
"""Summarize Agent-R1 TeleLogs validation trajectory dumps.

The trainer writes one JSON object per trajectory. Environment diagnostics are
attached to individual steps, with the structured diagnosis nested in the
tool_calls list of the terminal step. This module converts those raw traces
into task/tool metrics and a paired checkpoint comparison with deterministic
bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from recipes.telelogs.constants import ROOT_CAUSES, normalize_root_causes, set_f1

COMPARISON_FIELDS = (
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
    submissions = [
        info for info in _iter_tool_infos(entry)
        if "predicted_root_causes" in info
    ]
    return submissions[-1] if submissions else None


def _stable_case_key(entry: dict[str, Any]) -> str:
    """Return a checkpoint-stable case identifier.

    New dumps persist case_key directly. For historical dumps, reconstruct the
    same key from the immutable validation input and ground truth instead of
    using trajectory_uid, which may be regenerated at every validation.
    """
    if entry.get("case_key"):
        return str(entry["case_key"])
    payload = json.dumps(
        {"input": entry.get("input"), "gts": entry.get("gts")},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trajectory_row(entry: dict[str, Any]) -> dict[str, Any]:
    expected = normalize_root_causes(entry.get("gts"))
    submission = _terminal_submission(entry)
    predicted = (
        normalize_root_causes(submission.get("predicted_root_causes"))
        if submission else []
    )
    tool_infos = list(_iter_tool_infos(entry))
    query_tools = [
        info.get("tool")
        for info in tool_infos
        if isinstance(info.get("tool"), str)
        and str(info["tool"]).startswith("query_")
    ]
    failures = sum(bool(info.get("error")) for info in tool_infos)
    missing_submission = submission is None
    if missing_submission:
        failures += 1
    components = submission.get("score_components", {}) if submission else {}
    return {
        "trajectory_uid": str(entry.get("trajectory_uid", "")),
        "case_key": _stable_case_key(entry),
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
        "evidence_groundedness": float(
            components.get("evidence_groundedness", 0.0)
        ),
        "cause_evidence_support": float(
            components.get("cause_evidence_support", 0.0)
        ),
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
    micro_f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
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
        "tool_call_failure_rate": (
            sum(row["tool_failure_count"] for row in rows) / max(1, total_queries)
        ),
        "evidence_groundedness": mean(
            row["evidence_groundedness"] for row in rows
        ),
        "cause_evidence_support": mean(
            row["cause_evidence_support"] for row in rows
        ),
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


def summarize_step(
    entries: list[dict[str, Any]], step: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        trajectory_row(entry)
        for entry in entries
        if int(entry.get("step", -1)) == step
    ]
    return summarize_rows(rows), rows


def compare_summaries(
    baseline: dict[str, Any], final: dict[str, Any]
) -> dict[str, float]:
    return {
        field: float(final[field]) - float(baseline[field])
        for field in COMPARISON_FIELDS
    }


def _index_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("case_key", ""))
        if not key:
            raise ValueError("validation row is missing case_key")
        if key in result:
            raise ValueError(f"duplicate validation case_key: {key}")
        result[key] = row
    return result


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    fraction = pos - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def paired_bootstrap_delta(
    baseline_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap 95% CIs over the same validation cases."""
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    baseline_by_case = _index_by_case(baseline_rows)
    final_by_case = _index_by_case(final_rows)
    if set(baseline_by_case) != set(final_by_case):
        missing_final = sorted(set(baseline_by_case) - set(final_by_case))
        missing_base = sorted(set(final_by_case) - set(baseline_by_case))
        raise ValueError(
            "baseline/final validation cases differ: "
            f"missing_final={len(missing_final)}, "
            f"missing_baseline={len(missing_base)}"
        )

    keys = sorted(baseline_by_case)
    if not keys:
        raise ValueError("no paired validation cases")
    observed = compare_summaries(
        summarize_rows([baseline_by_case[key] for key in keys]),
        summarize_rows([final_by_case[key] for key in keys]),
    )
    draws: dict[str, list[float]] = {
        field: [] for field in COMPARISON_FIELDS
    }
    rng = random.Random(seed)
    for _ in range(resamples):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        baseline = summarize_rows([baseline_by_case[key] for key in sampled])
        final = summarize_rows([final_by_case[key] for key in sampled])
        delta = compare_summaries(baseline, final)
        for field in COMPARISON_FIELDS:
            draws[field].append(delta[field])

    return {
        field: {
            "estimate": observed[field],
            "lower_95": _percentile(draws[field], 0.025),
            "upper_95": _percentile(draws[field], 0.975),
        }
        for field in COMPARISON_FIELDS
    }


def markdown_report(
    baseline: dict[str, Any],
    final: dict[str, Any],
    delta: dict[str, float],
    confidence_intervals: dict[str, dict[str, float]] | None = None,
) -> str:
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
        "Step 0 is the frozen base-model baseline; the final step is the trained policy. "
        "Where available, the CI is a paired bootstrap over the same validation cases.",
        "",
        "| Metric | Step 0 | Final | Delta | 95% CI for delta |",
        "|---|---:|---:|---:|---:|",
    ]
    confidence_intervals = confidence_intervals or {}
    for field in fields:
        before = baseline[field]
        after = final[field]
        if field == "trajectories":
            lines.append(f"| {field} | {before} | {after} | — | — |")
            continue
        ci = confidence_intervals.get(field)
        rendered_ci = (
            f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}]"
            if ci else "—"
        )
        lines.append(
            f"| {field} | {before:.4f} | {after:.4f} | "
            f"{delta.get(field, 0.0):+.4f} | {rendered_ci} |"
        )
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
        lines.append(
            f"| {cause} | {samples} | {before_em:.4f} | "
            f"{after_em:.4f} | {change:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize TeleLogs Agent-R1 validation dumps"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Validation JSONL directory or file"
    )
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument(
        "--final-step", type=int, default=None,
        help="Defaults to the largest available step"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/telelogs_validation"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    entries = load_entries(args.input.expanduser().resolve())
    available_steps = sorted({int(entry.get("step", -1)) for entry in entries})
    final_step = (
        args.final_step if args.final_step is not None else available_steps[-1]
    )
    baseline, baseline_rows = summarize_step(entries, args.baseline_step)
    final, final_rows = summarize_step(entries, final_step)
    delta = compare_summaries(baseline, final)
    confidence_intervals = paired_bootstrap_delta(
        baseline_rows,
        final_rows,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    payload = {
        "source": str(args.input),
        "available_steps": available_steps,
        "baseline": baseline,
        "final": final,
        "delta": delta,
        "paired_bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence_intervals": confidence_intervals,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown_report(baseline, final, delta, confidence_intervals),
        encoding="utf-8",
    )
    with (args.output_dir / "trajectory_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in baseline_rows + final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
