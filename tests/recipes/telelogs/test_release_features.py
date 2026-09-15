import unittest
from agent_r1.env.base import Action
from recipes.telelogs.env.telelogs_env import TeleLogsEnv
from tests.recipes.telelogs.test_environment import tool_call


class ReleaseProtocolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = TeleLogsEnv(
            case={"sections": {"radio_kpi": ["SINR -5 dB", "RSRP -80 dBm"], "handover": ["handover success rate 60%"]}},
            ground_truth=["C4"],
        )
        self.env.reset()

    async def test_new_records_in_same_tool_can_earn_evidence_reward(self):
        await self.env.step(Action(text=tool_call("query_radio_kpi", {"focus": "SINR"})))
        obs, reward, done, info = await self.env.step(Action(text=tool_call("query_radio_kpi", {"focus": "RSRP"})))
        self.assertGreater(reward, 0)
        self.assertIn("evidence_id", obs.messages[-1]["content"])

    async def test_every_query_pays_a_cost(self):
        _, reward, _, info = await self.env.step(Action(text=tool_call("query_radio_kpi", {"focus": "absent"})))
        self.assertLess(reward, 0)

    async def test_invalid_submission_does_not_end_episode(self):
        for tool in ["query_radio_kpi", "query_handover"]:
            await self.env.step(Action(text=tool_call(tool, {})))
        _, reward, done, info = await self.env.step(Action(text=tool_call("submit_diagnosis", {"root_causes": ["C4"]})))
        self.assertFalse(done)
        self.assertLess(reward, 0)

    async def test_unknown_query_argument_is_rejected(self):
        _, reward, _, info = await self.env.step(Action(text=tool_call("query_radio_kpi", {"ground_truth": True})))
        self.assertLess(reward, 0)
        self.assertEqual(self.env._queried, [])
