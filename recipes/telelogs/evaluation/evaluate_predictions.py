"""Evaluate JSONL predictions on TeleLogs or TeleLogsAgent TS1/TS2/TS3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from recipes.telelogs.constants import normalize_root_causes, set_f1


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prediction file contains no rows")
    exact = 0
    macro_f1 = 0.0
    tool_calls = 0
    tool_failures = 0
    iterations = 0
    by_suite: dict[str, list[float]] = defaultdict(list)
    true_positive = false_positive = false_negative = 0

    for row in rows:
        predicted = set(normalize_root_causes(row.get("predicted_root_causes", row.get("prediction"))))
        expected = set(normalize_root_causes(row.get("ground_truth", row.get("target"))))
        if not expected:
            raise ValueError(f"row {row.get('scenario_id', '?')} has no valid ground truth")
        score = set_f1(predicted, expected)
        exact += int(predicted == expected)
        macro_f1 += score
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        calls = int(row.get("tool_call_count", len(row.get("tool_calls", []))))
        failures = int(row.get("tool_failure_count", 0))
        tool_calls += calls
        tool_failures += failures
        iterations += int(row.get("iterations", calls + 1))
        by_suite[str(row.get("suite", row.get("split", "unknown")))].append(score)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    micro_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    count = len(rows)
    return {
        "samples": count,
        "exact_match": exact / count,
        "mean_sample_set_f1": macro_f1 / count,
        "macro_f1": macro_f1 / count,  # legacy alias, not per-class macro F1
        "micro_f1": micro_f1,
        "avg_tool_calls": tool_calls / count,
        "tool_call_failure_rate": tool_failures / max(1, tool_calls),
        "avg_iterations": iterations / count,
        "accuracy_per_tool_call": exact / max(1, tool_calls),
        "suite_macro_f1": {suite: sum(scores) / len(scores) for suite, scores in sorted(by_suite.items())},
    }


def _markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# TeleLogs evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "samples",
        "exact_match",
        "macro_f1",
        "micro_f1",
        "avg_tool_calls",
        "tool_call_failure_rate",
        "avg_iterations",
        "accuracy_per_tool_call",
    ):
        value = metrics[key]
        rendered = str(value) if key == "samples" else f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.extend(["", "## Per suite", "", "| Suite | Macro F1 |", "|---|---:|"])
    for suite, score in metrics["suite_macro_f1"].items():
        lines.append(f"| {suite} | {score:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/telelogs_eval"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = evaluate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
