"""Fixed synthetic development cases, including unseen cause combinations."""

import argparse, json
from pathlib import Path
from scripts.telelogs.make_synthetic_cases import cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    rows = list(cases())
    lookup = {r["ground_truth"][0]: r for r in rows}
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
    for first, second in [("C1", "C6"), ("C2", "C8"), ("C3", "C7"), ("C5", "C6")]:
        row = json.loads(json.dumps(lookup[first]))
        row.update(id=f"compound-dev-{first}-{second}", ground_truth=[first, second], split="synthetic_compound_dev")
        row["case"]["sections"][views[second]] = lookup[second]["case"]["sections"][views[second]]
        rows.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(x) + "\n" for x in rows))


if __name__ == "__main__":
    main()
