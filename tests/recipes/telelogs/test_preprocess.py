import json
import unittest

from recipes.telelogs.data_preprocess.process_telelogs import convert_record


class TeleLogsPreprocessTest(unittest.TestCase):
    def test_conversion_hides_full_context_in_environment(self):
        record = {
            "question": "Observed symptom: throughput is 500 Mbps.\nVehicle speed is 52 km/h.\nRSRP is -90 dBm.",
            "answer": "C1",
            "scenario_id": "case-1",
        }
        converted = convert_record(record, 0, "train", "netop/TeleLogs")
        self.assertEqual(converted["agent_name"], "telelogs_rca")
        self.assertNotIn("52 km/h", converted["prompt"][1]["content"])
        env_kwargs = json.loads(converted["env_kwargs"])
        self.assertIn("52 km/h", env_kwargs["case"]["case_document"])
        self.assertEqual(env_kwargs["ground_truth"], ["C1"])


if __name__ == "__main__":
    unittest.main()
