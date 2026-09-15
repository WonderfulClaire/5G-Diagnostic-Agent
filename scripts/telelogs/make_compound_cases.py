"""Training-only compositional curriculum from existing synthetic train scenarios."""

import argparse, json
from pathlib import Path
from scripts.telelogs.make_training_cases import training_cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    source = list(training_cases())
    by_cause = {r["ground_truth"][0]: r for r in source[:8]}
    views = {
        "C1": "mobility",
        "C2": "antenna",
        "C3": "cell_relation",
        "C4": "cell_relation",
        "C5": "cell_relation",
        "C6": "handover",
        "C7": "handover",
        "C8": "resource",
    }
    rows = []
    for first, second in [
        ("C1", "C8"),
        ("C2", "C6"),
        ("C4", "C7"),
        ("C5", "C8"),
        ("C1", "C2"),
        ("C3", "C6"),
        ("C4", "C8"),
        ("C2", "C7"),
    ]:
        row = json.loads(json.dumps(by_cause[first]))
        other = by_cause[second]
        row.update(
            id=f"compound-train-{first}-{second}",
            ground_truth=[first, second],
            source="synthetic_compositional_curriculum_v1",
            parent_ids=[row["id"], other["id"]],
        )
        row["case"]["sections"][views[second]] = other["case"]["sections"][views[second]]
        rows.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(x) + "\n" for x in rows))


if __name__ == "__main__":
    main()
