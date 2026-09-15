import math
import unittest

from agent_r1.trainer.ppo.validation_metrics import build_numeric_validation_metrics


class ValidationMetricsTest(unittest.TestCase):
    def test_keeps_aligned_numeric_leaves_and_drops_rich_diagnostics(self):
        metrics = build_numeric_validation_metrics(
            {
                "reward": [0.5, 1.0],
                "components": [
                    {"root_cause": 1, "evidence": 0.25, "label": "C1"},
                    {"root_cause": 0, "evidence": 0.75, "label": "C2"},
                ],
                "tool_trace": [["query_radio_kpi"], ["query_handover"]],
                "diagnosis": [{"causes": ["C1"]}, {"causes": ["C2"]}],
            }
        )

        self.assertEqual(metrics["reward"], [0.5, 1.0])
        self.assertEqual(metrics["components.root_cause"], [1.0, 0.0])
        self.assertEqual(metrics["components.evidence"], [0.25, 0.75])
        self.assertNotIn("components.label", metrics)
        self.assertNotIn("tool_trace", metrics)
        self.assertNotIn("diagnosis.causes", metrics)

    def test_drops_partial_and_non_finite_metrics(self):
        metrics = build_numeric_validation_metrics(
            {
                "reward": [0.1, 0.2],
                "optional": [{"score": 1.0}, {}],
                "unstable": [1.0, math.nan],
            }
        )

        self.assertEqual(metrics, {"reward": [0.1, 0.2]})

    def test_rejects_misaligned_fields(self):
        with self.assertRaisesRegex(ValueError, "must be aligned"):
            build_numeric_validation_metrics({"reward": [1.0, 2.0], "score": [1.0]})


if __name__ == "__main__":
    unittest.main()
