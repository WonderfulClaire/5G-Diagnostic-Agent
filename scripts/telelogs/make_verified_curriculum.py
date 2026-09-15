"""Training-only measured fixtures with independent labels and no answer IDs.

This is a controlled synthetic curriculum, not a telecom benchmark. Values and
normal observations vary while all eight diagnostic dimensions stay observable.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

PAIRS = [("C1", "C8"), ("C5", "C8"), ("C3", "C5"), ("C3", "C4"),
         ("C4", "C7"), ("C6", "C7"), ("C2", "C6"), ("C1", "C2")]


def verified_labels(m):
    flags = [m["speed"] > 40, m["tilt"] > 12 and m["weak_far_edge"],
             m["distance"] > 1 and m["rsrp"] < -105,
             m["same_carrier"] and not m["colocated"] and m["interference"],
             m["pci"] % 30 == m["neighbor_pci"] % 30,
             m["handovers"] >= 10 and m["pingpong"], m["a3"] > 10,
             m["blocks"] < 160]
    return [f"C{i+1}" for i, flag in enumerate(flags) if flag]


def curriculum(variants=32, seed=20260915):
    for variant in range(variants):
        for causes in [(f"C{i}",) for i in range(1, 9)] + PAIRS:
            # Reset the generator for each combination: nuisance readings and
            # symptoms have the same distribution for every label combination.
            r = random.Random(seed + variant)
            m = {"speed": r.randint(12, 38), "tilt": r.randint(2, 7),
                 "weak_far_edge": False, "distance": round(r.uniform(.2, .8), 2),
                 "rsrp": r.randint(-95, -80), "same_carrier": False,
                 "colocated": False, "interference": False, "pci": r.randint(1, 400),
                 "handovers": r.randint(0, 2), "pingpong": False,
                 "a3": r.randint(2, 4), "blocks": r.randint(175, 230)}
            m["neighbor_pci"] = m["pci"] + 1
            abnormal = {"speed": r.randint(43, 77), "tilt": r.randint(14, 25),
                        "distance": round(r.uniform(1.15, 2.5), 2), "rsrp": r.randint(-124, -110),
                        "handovers": r.randint(12, 28), "a3": r.randint(13, 25),
                        "blocks": r.randint(85, 155)}
            if "C1" in causes: m["speed"] = abnormal["speed"]
            if "C2" in causes: m.update(tilt=abnormal["tilt"], weak_far_edge=True)
            if "C3" in causes: m.update(distance=abnormal["distance"], rsrp=abnormal["rsrp"])
            if "C4" in causes: m.update(same_carrier=True, interference=True)
            if "C5" in causes: m["neighbor_pci"] = m["pci"] + 30
            if "C6" in causes: m.update(handovers=abnormal["handovers"], pingpong=True)
            if "C7" in causes: m["a3"] = abnormal["a3"]
            if "C8" in causes: m["blocks"] = abnormal["blocks"]
            throughput = r.randint(340, 550)
            sections = {
                "radio_kpi": [f"Measured downlink throughput is {throughput} Mbps."],
                "mobility": [f"Test vehicle speed: {m['speed']} km/h. Maximum permitted speed: 40 km/h."],
                "antenna": [f"Serving antenna downtilt: {m['tilt']} degrees. " +
                            ("Excessive tilt weakens far-end coverage." if m["weak_far_edge"] else "Far-end coverage is adequate.")],
                "cell_relation": [f"Serving-cell distance: {m['distance']} km; serving RSRP: {m['rsrp']} dBm.",
                                  "The non-colocated neighbor " + ("uses the same carrier and causes severe interference." if m["interference"] else "uses a different carrier; interference is absent."),
                                  f"Serving PCI: {m['pci']}; neighbor PCI: {m['neighbor_pci']}."] ,
                "handover": [f"Handover events per minute: {m['handovers']}. " + ("Repeated ping-pong returns to the previous cell." if m["pingpong"] else "No ping-pong events."),
                             f"A3 threshold: {m['a3']} dB. " + ("The threshold is misconfigured and blocks transfer to a stronger neighbor." if "C7" in causes else "The threshold is correctly configured.")],
                "resource": [f"Average scheduled resource blocks: {m['blocks']}; required minimum: 160."],
            }
            assert verified_labels(m) == sorted(causes)
            content = json.dumps(sections, sort_keys=True)
            yield {"id": hashlib.sha256(content.encode()).hexdigest()[:20], "split": "train",
                   "source": "verified_numeric_curriculum_v1", "variant_group": variant,
                   "symptom": f"Downlink throughput is {throughput} Mbps. Identify all supported faults.",
                   "case": {"sections": sections}, "ground_truth": sorted(causes),
                   "audit_measurements": m, "rule_version": "numeric-fixture-v1"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--variants", type=int, default=32)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = list(curriculum(a.variants))
    manifest = {}
    for name, selected in [("single", [r for r in rows if len(r["ground_truth"]) == 1]), ("mixed", rows)]:
        payload = "".join(json.dumps(r) + "\n" for r in selected).encode()
        (a.output / (name + ".jsonl")).write_bytes(payload)
        manifest[name] = {"rows": len(selected), "sha256": hashlib.sha256(payload).hexdigest()}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
