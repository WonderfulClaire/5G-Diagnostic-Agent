"""Disjoint synthetic train scenarios; same taxonomy, distinct wording and readings.

These fixtures test learning mechanics. They are not a telecom benchmark or an
unseen-domain evaluation, and have deliberately simple causal labels.
"""

import argparse, json
from pathlib import Path
from scripts.telelogs.make_synthetic_cases import cases


def training_cases():
    for variant in range(4):
        for index, source in enumerate(cases()):
            cause = source["ground_truth"][0]
            source["id"] = f"synthetic-train-{variant}-{cause}"
            source["split"] = "train"
            source["symptom"] = (
                f"Drive-test session {variant}-{index}: user download throughput is {380 + variant * 11} Mbps. Find the supported fault."
            )
            section = {
                "C1": "mobility",
                "C2": "antenna",
                "C3": "cell_relation",
                "C4": "cell_relation",
                "C5": "cell_relation",
                "C6": "handover",
                "C7": "handover",
                "C8": "resource",
            }[cause]
            text = {
                "C1": f"GPS measurement: vehicle travels at {52 + variant * 3} km/h, above the 40 km/h limit.",
                "C2": f"The antenna vertical tilt is configured at {16 + variant} degrees. This excessive downtilt weakens far-end coverage.",
                "C3": f"UE-to-serving-site separation is {1.3 + variant * 0.1:.1f} km; RSRP is -115 dBm. Coverage distance exceeds 1 km.",
                "C4": f"Strong interfering neighbor at RSRP -{74 + variant} dBm uses the same frequency but is located at a different site.",
                "C5": f"PCI planning issue: serving PCI {32 + variant} and neighbor PCI {62 + variant} give identical remainders modulo 30.",
                "C6": f"The event log counts {12 + variant} handovers per minute, repeatedly returning to the prior cell (ping-pong).",
                "C7": f"The handover A3 threshold is set to {14 + variant} dB. This incorrect threshold prevents transfer to a stronger neighbor.",
                "C8": f"Scheduler statistics show {110 + variant * 7} resource blocks on average; required average allocation is at least 160.",
            }[cause]
            source["case"]["sections"][section] = [text]
            source["case"]["sections"]["radio_kpi"] = [f"Downlink throughput: {380 + variant * 11} Mbps."]
            source["source"] = "synthetic_training_templates_v1"
            yield source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(x) + "\n" for x in training_cases()))


if __name__ == "__main__":
    main()
