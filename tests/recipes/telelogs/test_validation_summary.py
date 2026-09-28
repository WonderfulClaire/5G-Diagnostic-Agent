import unittest

from recipes.telelogs.evaluation.summarize_validation import (
    compare_summaries,
    paired_bootstrap_delta,
    summarize_rows,
    trajectory_row,
)


def _entry(
    *,
    step,
    predicted,
    expected="C4",
    case="case-a",
    query_tools=("query_radio_kpi", "query_cell_relation"),
):
    tool_steps = [
        {
            "step_index": index,
            "tool_calls": [{"tool": tool, "record_count": 1}],
        }
        for index, tool in enumerate(query_tools)
    ]
    if predicted is not None:
        tool_steps.append(
            {
                "step_index": len(tool_steps),
                "tool_calls": [
                    {
                        "predicted_root_causes": predicted,
                        "score_components": {
                            "evidence_groundedness": 1.0,
                            "cause_evidence_support": 1.0,
                            "repair_relevance": 0.5,
                            "tool_efficiency": 1.0,
                        },
                    }
                ],
            }
        )
    return {
        # trajectory_uid may legitimately change between validation checkpoints.
        "trajectory_uid": f"rollout-{step}-{case}",
        "input": f"symptom for {case}",
        "step": step,
        "gts": expected,
        "score": 0.9 if predicted == expected else 0.1,
        "num_steps": len(tool_steps),
        "steps": tool_steps,
    }


class ValidationSummaryTest(unittest.TestCase):
    def test_extracts_terminal_submission_and_tool_metrics(self):
        row = trajectory_row(_entry(step=300, predicted="C4"))
        self.assertEqual(row["predicted_root_causes"], ["C4"])
        self.assertEqual(row["query_count"], 2)
        self.assertEqual(row["exact"], 1.0)
        self.assertEqual(row["repair_relevance"], 0.5)
        self.assertTrue(row["case_key"])

    def test_missing_submission_is_counted_as_failure(self):
        row = trajectory_row(_entry(step=0, predicted=None))
        self.assertEqual(row["submitted"], 0.0)
        self.assertEqual(row["tool_failure_count"], 1)
        self.assertEqual(row["root_cause_f1"], 0.0)

    def test_summary_and_delta(self):
        baseline_rows = [
            trajectory_row(_entry(step=0, predicted="C5", case="case-a")),
            trajectory_row(_entry(step=0, predicted="C4", case="case-b")),
        ]
        final_rows = [
            trajectory_row(_entry(step=300, predicted="C4", case="case-a")),
            trajectory_row(_entry(step=300, predicted="C4", case="case-b")),
        ]
        baseline = summarize_rows(baseline_rows)
        final = summarize_rows(final_rows)
        delta = compare_summaries(baseline, final)
        self.assertEqual(baseline["exact_match"], 0.5)
        self.assertEqual(final["exact_match"], 1.0)
        self.assertEqual(delta["exact_match"], 0.5)

    def test_paired_bootstrap_uses_stable_case_key_not_rollout_uid(self):
        baseline_rows = [
            trajectory_row(_entry(step=0, predicted="C5", case="case-a")),
            trajectory_row(_entry(step=0, predicted="C5", case="case-b")),
            trajectory_row(_entry(step=0, predicted="C4", case="case-c")),
            trajectory_row(_entry(step=0, predicted="C4", case="case-d")),
        ]
        final_rows = [
            trajectory_row(_entry(step=300, predicted="C4", case="case-a")),
            trajectory_row(_entry(step=300, predicted="C4", case="case-b")),
            trajectory_row(_entry(step=300, predicted="C4", case="case-c")),
            trajectory_row(_entry(step=300, predicted="C4", case="case-d")),
        ]
        result = paired_bootstrap_delta(
            baseline_rows, final_rows, resamples=200, seed=7
        )
        self.assertEqual(result["exact_match"]["estimate"], 0.5)
        self.assertLessEqual(
            result["exact_match"]["lower_95"],
            result["exact_match"]["estimate"],
        )
        self.assertGreaterEqual(
            result["exact_match"]["upper_95"],
            result["exact_match"]["estimate"],
        )


if __name__ == "__main__":
    unittest.main()
