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


if __name__ == "__main__":
    unittest.main()
