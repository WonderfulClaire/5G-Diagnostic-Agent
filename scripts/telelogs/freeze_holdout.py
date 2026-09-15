"""Create an immutable synthetic holdout before the next training experiment.

New wording and readings test limited template transfer, not real-world validity.
Pair combinations exclude both existing training and inspected development pairs.
The labels remain explicit heuristic fixtures, never telecom ground truth.
"""

import argparse
import hashlib
import json
from pathlib import Path

VIEWS = dict(zip((f"C{i}" for i in range(1, 9)),
                 ("mobility", "antenna", "cell_relation", "cell_relation",
                  "cell_relation", "handover", "handover", "resource")))
USED_PAIRS = {
    ("C1", "C8"), ("C2", "C6"), ("C4", "C7"), ("C5", "C8"),
    ("C1", "C2"), ("C3", "C6"), ("C4", "C8"), ("C2", "C7"),
    ("C1", "C6"), ("C2", "C8"), ("C3", "C7"), ("C5", "C6"),
}


def heldout_cases():
    # Balanced single/pair counts. The pair list is fixed, never selected by scores.
    pairs = [("C1", "C3"), ("C1", "C7"), ("C2", "C4"), ("C2", "C5"),
             ("C3", "C8"), ("C4", "C6"), ("C5", "C7"), ("C6", "C8")]
    groups = [(c,) for c in VIEWS] + pairs
    for variant in range(2):
        for causes in groups:
            sections = {
                "radio_kpi": [f"Measured downlink rate: {347 + 19 * variant} Mbps."],
                "mobility": ["Drive speed remains 22 km/h, within the allowed 40 km/h."],
                "antenna": ["Downtilt is 5 degrees and the coverage footprint is adequate."],
                "cell_relation": ["Serving site is 0.3 km away; no frequency interference or PCI conflict is found."],
                "handover": ["Handover events are rare; the A3 configuration passes inspection."],
                "resource": ["The scheduler allocates 205 blocks on average, exceeding the 160-block requirement."],
            }
            abnormal = {
                "C1": f"During the slow transfer the vehicle is travelling at {71 + variant} km/h. The permitted maximum is 40 km/h.",
                "C2": f"Inspection finds a {21 + variant}-degree downward antenna angle, too steep to cover the far edge.",
                "C3": f"The receiver is {2.2 + variant * .1:.1f} km from its serving site with -121 dBm RSRP; the coverage-distance limit is 1 km.",
                "C4": f"A neighbor on another physical site transmits on the serving carrier at -{69 + variant} dBm and interferes with reception.",
                "C5": f"Serving PCI {127 + variant} and strong neighbor PCI {157 + variant} have the same remainder after division by 30.",
                "C6": f"Within 60 seconds the UE switches cells {22 + variant} times, repeatedly switching back to the previous cell.",
                "C7": f"A3 is wrongly configured to {23 + variant} dB, blocking handover despite a stronger available neighbor.",
                "C8": f"Only {81 + variant} resource blocks are allocated on average against a minimum requirement of 160.",
            }
            for view in {VIEWS[c] for c in causes}:
                sections[view] = [abnormal[c] for c in causes if VIEWS[c] == view]
            group = "+".join(causes)
            yield {
                "id": f"frozen-v1-{variant}-{group}", "split": "test",
                "group_id": f"frozen-v1-{group}",
                "source": "synthetic_holdout_v1_new_wording",
                "symptom": "A drive-test transfer is unexpectedly slow. Identify every fault supported by the diagnostic observations.",
                "case": {"sections": sections}, "ground_truth": list(causes),
            }


def freeze(output):
    rows = list(heldout_cases())
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    # A second invocation cannot silently replace the evaluated dataset.
    output.mkdir(parents=True, exist_ok=False)
    (output / "test.jsonl").write_bytes(payload)
    manifest = {
        "sha256": hashlib.sha256(payload).hexdigest(), "samples": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "excluded_pairs": sorted(USED_PAIRS),
        "scope": "synthetic wording and combination transfer; not independent real-world data",
        "policy": "Never mine, review, train on, or select hyperparameters using this test. After inspection it is a regression set for later rounds.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    print(json.dumps(freeze(parser.parse_args().output), indent=2))
