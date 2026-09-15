from collections import Counter
from scripts.telelogs.make_verified_curriculum import curriculum, PAIRS, verified_labels
from scripts.telelogs.freeze_holdout import heldout_cases


def test_training_curriculum_labels_balance_and_frozen_pair_separation():
    rows = list(curriculum(2))
    assert len(rows) == len({r['id'] for r in rows}) == 32
    assert all(verified_labels(r['audit_measurements']) == r['ground_truth'] for r in rows)
    assert len({r['symptom'] for r in rows if r['variant_group'] == 0}) == 1
    assert set(Counter(c for p in PAIRS for c in p).values()) == {2}
    held = {tuple(r['ground_truth']) for r in heldout_cases() if len(r['ground_truth']) == 2}
    assert not set(PAIRS) & held
    assert all(r['split'] == 'train' for r in rows)
