import pytest
from recipes.telelogs.training.token_weights import diagnosis_token_weights


def test_weights_only_overlap_decision_value_not_evidence_or_protocol():
    text = '{"root_causes": ["C1", "C8"], "evidence": ["C1 mentioned"]}'
    offsets = [(i, i+1) for i in range(len(text))] + [(0, 0)]
    weights = diagnosis_token_weights(text, offsets, 8)
    weighted_text = ''.join(c for c, w in zip(text, weights) if w == 8)
    assert weighted_text == '["C1", "C8"]'
    assert weights[-1] == 1  # tokenizer special token
    assert diagnosis_token_weights(text, offsets) == [1] * len(offsets)


def test_boundary_tokens_and_queries():
    text = '{"root_causes": ["C1"]}'
    start = text.index('[')
    assert diagnosis_token_weights(text, [(start-1, start+2)], 4) == [4]
    assert diagnosis_token_weights('{"name":"query_mobility"}', [(0, 8)], 4) == [1]
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            diagnosis_token_weights(text, [], bad)
