import unittest

import numpy as np

from agent_r1.agent_flow.agent_flow import _align_non_tensor_batch_keys, _to_object_array


class FakeDataProto:
    def __init__(self, non_tensor_batch, length, reward_extra_keys=None):
        self.non_tensor_batch = non_tensor_batch
        self.length = length
        self.meta_info = {"reward_extra_keys": reward_extra_keys or []}

    def __len__(self):
        return self.length


class AgentFlowMetadataTest(unittest.TestCase):
    def test_ragged_reward_metadata_is_preserved(self):
        values = [[{"tool": "query_radio_kpi"}], [], [{"tool": "query_handover"}, {"tool": "submit_diagnosis"}]]

        result = _to_object_array(values)

        self.assertEqual(result.dtype, np.dtype(object))
        self.assertEqual(result.tolist(), values)

    def test_distributed_metadata_keys_are_aligned(self):
        left = FakeDataProto(
            {"tool": _to_object_array(["query_radio_kpi"])}, 1, reward_extra_keys=["tool"]
        )
        right = FakeDataProto(
            {"score": _to_object_array([0.8, 0.9])}, 2, reward_extra_keys=["score"]
        )

        _align_non_tensor_batch_keys([left, right])

        self.assertEqual(set(left.non_tensor_batch), {"tool", "score"})
        self.assertEqual(set(right.non_tensor_batch), {"tool", "score"})
        self.assertEqual(left.non_tensor_batch["score"].tolist(), [None])
        self.assertEqual(right.non_tensor_batch["tool"].tolist(), [None, None])
        self.assertEqual(left.meta_info["reward_extra_keys"], ["score", "tool"])
        self.assertEqual(right.meta_info["reward_extra_keys"], ["score", "tool"])


if __name__ == "__main__":
    unittest.main()
