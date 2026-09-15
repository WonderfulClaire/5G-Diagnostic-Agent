import hashlib
import tempfile
from pathlib import Path

import pytest

from scripts.telelogs.freeze_holdout import USED_PAIRS, freeze, heldout_cases


def test_holdout_is_disjoint_by_pair_and_immutable():
    rows = list(heldout_cases())
    assert len(rows) == len({r["id"] for r in rows}) == 32
    assert all(r["split"] == "test" for r in rows)
    assert all(tuple(r["ground_truth"]) not in USED_PAIRS for r in rows)
    assert sum(len(r["ground_truth"]) == 1 for r in rows) == 16
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "frozen"
        manifest = freeze(dest)
        assert manifest["sha256"] == hashlib.sha256((dest / "test.jsonl").read_bytes()).hexdigest()
        with pytest.raises(FileExistsError):
            freeze(dest)
