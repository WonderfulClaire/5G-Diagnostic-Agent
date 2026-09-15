"""Generate explicit synthetic protocol fixtures; not official TeleLogs examples."""

import argparse
import json
from pathlib import Path

SCENARIOS = [
    ("C1", "mobility", "Vehicle speed is 65 km/h throughout the throughput drop."),
    ("C2", "antenna", "Serving antenna downtilt is excessive at 18 degrees; far-end coverage becomes weak."),
    ("C3", "cell_relation", "Serving-cell distance is 1.8 km and serving RSRP is -118 dBm."),
    ("C4", "cell_relation", "A non-colocated co-frequency neighbor causes interference; neighbor RSRP is -76 dBm."),
    ("C5", "cell_relation", "Serving PCI is 31 and strong neighbor PCI is 61; their PCI modulo 30 values collide."),
    ("C6", "handover", "Frequent ping-pong handovers: handover count is 18 in one minute."),
    ("C7", "handover", "Handover threshold A3 is misconfigured at 18 dB; a stronger neighbor is never selected."),
    ("C8", "resource", "Average scheduled resource blocks are 95, below the 160-block requirement."),
]


def cases():
    for i, (cause, view, evidence) in enumerate(SCENARIOS):
        sections = {
            "radio_kpi": ["Observed downlink throughput is 410 Mbps."],
            "mobility": ["Vehicle speed is 25 km/h."],
            "antenna": ["Serving antenna downtilt is 4 degrees; coverage is adequate."],
            "cell_relation": ["Serving distance is 0.4 km. No co-frequency interference or PCI collision."],
            "handover": ["One successful handover in 10 minutes; thresholds are correctly configured."],
            "resource": ["Average scheduled resource blocks are 210."],
        }
        sections[view] = [evidence]
        if cause == "C4":
            sections["radio_kpi"].append("SINR is -7 dB despite strong serving RSRP -80 dBm.")
        if cause == "C3":
            sections["radio_kpi"].append("RSRP is -118 dBm.")
        yield {
            "id": f"synthetic-protocol-{i:03d}",
            "split": "synthetic_dev",
            "source": "hand_authored_fixture_v1",
            "symptom": "Downlink throughput fell to 410 Mbps. Query evidence to identify the cause.",
            "case": {"sections": sections},
            "ground_truth": [cause],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(row) + "\n" for row in cases()))


if __name__ == "__main__":
    main()
