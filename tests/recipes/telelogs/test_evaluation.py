import unittest

from recipes.telelogs.evaluation.evaluate_predictions import evaluate_rows


class EvaluationTest(unittest.TestCase):
    def test_metrics(self):
        metrics = evaluate_rows(
            [
                {
                    "suite": "TS1",
                    "ground_truth": ["C4"],
                    "predicted_root_causes": ["C4"],
                    "tool_call_count": 3,
                    "iterations": 4,
                },
                {
                    "suite": "TS2",
                    "ground_truth": ["C5"],
                    "predicted_root_causes": ["C4"],
                    "tool_call_count": 2,
                    "tool_failure_count": 1,
                    "iterations": 3,
                },
            ]
        )
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertAlmostEqual(metrics["tool_call_failure_rate"], 0.2)

    def test_equal_overall_accuracy_exposes_retention_regression(self):
        before = evaluate_rows([
            {"ground_truth": ["C1"], "prediction": ["C1"]},
            {"ground_truth": ["C2", "C8"], "prediction": ["C2"]},
        ])
        after = evaluate_rows([
            {"ground_truth": ["C1"], "prediction": ["C1", "C8"]},
            {"ground_truth": ["C2", "C8"], "prediction": ["C2", "C8"]},
        ])
        self.assertEqual(before["exact_match"], after["exact_match"])
        self.assertEqual(after["by_root_cause_count"]["1"]["exact_match"], 0)
        self.assertEqual(after["by_root_cause_count"]["1"]["overprediction_rate"], 1)
        self.assertEqual(before["by_root_cause_count"]["2"]["underprediction_rate"], 1)


if __name__ == "__main__":
    unittest.main()
