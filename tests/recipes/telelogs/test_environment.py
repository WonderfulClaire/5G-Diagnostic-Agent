import json
import unittest

from agent_r1.env.base import Action
from recipes.telelogs.env.telelogs_env import TeleLogsEnv, split_case_document


def tool_call(name, arguments):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


class TeleLogsEnvironmentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        document = """Observed symptom: throughput is 410 Mbps.
Serving-cell RSRP is -84 dBm while SINR is -5 dB.
Neighbor PCI 72 is co-frequency with serving PCI 11 and is not colocated.
Vehicle speed is 25 km/h.
Antenna downtilt is 4 degrees.
Average scheduled RBs are 210.
"""
        self.env = TeleLogsEnv(
            case={"case_document": document, "sections": split_case_document(document)},
            ground_truth=["C4"],
        )
        self.env.reset(raw_prompt=[{"role": "user", "content": "diagnose"}])

    async def test_query_then_correct_submission(self):
        observation, reward, done, info = await self.env.step(Action(text=tool_call("query_radio_kpi", {})))
        self.assertFalse(done)
        self.assertGreater(reward, 0)
        self.assertIn("SINR", observation.messages[-1]["content"])

        observation, reward, done, info = await self.env.step(Action(text=tool_call("query_cell_relation", {})))
        self.assertFalse(done)
        self.assertIn("co-frequency", observation.messages[-1]["content"])

        submission = {
            "root_causes": ["C4"],
            "evidence": ["SINR is -5 dB and a non-colocated co-frequency neighbor is present"],
            "repair_actions": ["Apply interference coordination and revise the frequency planning"],
            "confidence": 0.92,
        }
        _, reward, done, info = await self.env.step(Action(text=tool_call("submit_diagnosis", submission)))
        self.assertTrue(done)
        self.assertGreaterEqual(reward, 0.85)
        self.assertEqual(info["tool_calls"][0]["predicted_root_causes"], ["C4"])

    async def test_repeated_query_is_penalized(self):
        await self.env.step(Action(text=tool_call("query_radio_kpi", {})))
        _, reward, done, _ = await self.env.step(Action(text=tool_call("query_radio_kpi", {})))
        self.assertFalse(done)
        self.assertLess(reward, 0)

    async def test_query_and_submit_in_same_generation_is_rejected_atomically(self):
        action = "\n".join(
            [
                tool_call("query_radio_kpi", {}),
                tool_call("query_cell_relation", {}),
                tool_call("submit_diagnosis", {"root_causes": ["C4"]}),
            ]
        )
        _, reward, done, info = await self.env.step(Action(text=action))
        self.assertFalse(done)
        self.assertLess(reward, 0)
        self.assertEqual(info["tool_calls"][0]["error"], "submit_must_be_separate_turn")
        self.assertEqual(self.env._queried, [])
        self.assertEqual(self.env._observed_views, set())

    async def test_empty_queries_cannot_unlock_submission_or_earn_reward(self):
        for name in ("query_radio_kpi", "query_cell_relation"):
            _, reward, _, _ = await self.env.step(Action(text=tool_call(name, {"focus": "no-such-record"})))
            self.assertEqual(reward, -self.env.query_cost)
        _, reward, done, info = await self.env.step(Action(text=tool_call("submit_diagnosis", {"root_causes": ["C4"]})))
        self.assertFalse(done)
        self.assertLess(reward, 0)
        self.assertEqual(info["tool_calls"][0]["error"], "insufficient_observed_evidence")

    async def test_batched_queries_then_later_submission_and_reset(self):
        calls = tool_call("query_radio_kpi", {}) + tool_call("query_cell_relation", {})
        _, _, done, _ = await self.env.step(Action(text=calls))
        self.assertFalse(done)
        submission = {
            "root_causes": ["C4"],
            "evidence": ["SINR -5 dB"],
            "repair_actions": ["interference coordination"],
            "confidence": 0.9,
        }
        _, _, done, info = await self.env.step(Action(text=tool_call("submit_diagnosis", submission)))
        self.assertTrue(done)
        self.assertEqual(info["tool_calls"][0]["predicted_root_causes"], ["C4"])
        self.env.reset(raw_prompt=[])
        _, _, done, info = await self.env.step(Action(text=tool_call("submit_diagnosis", {"root_causes": ["C4"]})))
        self.assertFalse(done)
        self.assertEqual(info["tool_calls"][0]["error"], "insufficient_observed_evidence")

    async def test_free_text_without_submission_terminates(self):
        _, reward, done, info = await self.env.step(Action(text="The answer is C4"))
        self.assertTrue(done)
        self.assertLess(reward, 0)
        self.assertEqual(info["error"], "missing_submit_diagnosis")

    def test_pipe_table_is_projected_and_summarized(self):
        document = """Timestamp|GPS Speed (km/h)|Serving PCI|Serving SS-SINR [dB]|DL Throughput [Mbps]|Rbs
2026-01-01 00:00:00|20|100|10|610|170
2026-01-01 00:00:01|52|100|-4|410|140
2026-01-01 00:00:02|48|130|-6|390|130
"""
        sections = split_case_document(document)
        mobility_stats = sections["mobility"][0]["statistics"]
        radio_stats = sections["radio_kpi"][0]["statistics"]
        resource_stats = sections["resource"][0]["statistics"]
        self.assertEqual(mobility_stats["GPS Speed (km/h)"]["max"], 52.0)
        self.assertEqual(radio_stats["Serving PCI"]["transitions"], 1)
        self.assertEqual(resource_stats["Rbs"]["mean"], 146.666667)
        self.assertNotIn("unique", mobility_stats["GPS Speed (km/h)"])
        self.assertEqual(len(sections["radio_kpi"][0]["sample_rows"]), 2)


if __name__ == "__main__":
    unittest.main()
