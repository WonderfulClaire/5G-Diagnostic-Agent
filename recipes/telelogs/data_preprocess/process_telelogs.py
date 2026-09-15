"""Convert gated TeleLogs rows into Agent-R1 multi-turn training parquet.

No benchmark samples are committed or redistributed.  Users must accept the
official Hugging Face dataset conditions and supply their own ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from recipes.telelogs.constants import normalize_root_causes
from recipes.telelogs.env.telelogs_env import extract_symptom, split_case_document
from recipes.telelogs.prompts import build_agent_messages

QUESTION_FIELDS = ("question", "prompt", "input", "instruction", "text", "case")
ANSWER_FIELDS = ("root_causes", "root_cause", "answer", "answers", "label", "labels", "target", "output")


def _first_present(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in record and record[field] not in (None, "", []):
            return record[field]
    return None


def _choices(record: dict[str, Any]) -> list[str] | None:
    value = record.get("choices", record.get("options"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, dict):
        value = list(value.values())
    return [str(item) for item in value] if isinstance(value, list) else None


def convert_record(record: dict[str, Any], index: int, split: str, data_source: str) -> dict[str, Any]:
    question_value = _first_present(record, QUESTION_FIELDS)
    if question_value is None:
        raise ValueError(f"Sample {split}/{index} has none of the supported question fields: {QUESTION_FIELDS}")
    question = str(question_value)
    answer_value = _first_present(record, ANSWER_FIELDS)
    ground_truth = normalize_root_causes(answer_value, choices=_choices(record))
    if not ground_truth:
        raise ValueError(f"Sample {split}/{index} has an unsupported TeleLogs answer: {answer_value!r}")

    scenario_id = str(record.get("scenario_id", record.get("id", f"{split}-{index:06d}")))
    symptom = extract_symptom(question)
    case = {
        "scenario_id": scenario_id,
        "symptom": symptom,
        "case_document": question,
        "sections": split_case_document(question),
    }
    ground_truth_text = ",".join(ground_truth)
    return {
        "data_source": data_source,
        "agent_name": "telelogs_rca",
        "prompt": build_agent_messages(scenario_id, symptom),
        "question": symptom,
        "ground_truth": ground_truth_text,
        "ability": "5g_root_cause_analysis",
        "reward_model": {"style": "rule", "ground_truth": ground_truth_text},
        "extra_info": {
            "split": split,
            "index": index,
            "scenario_id": scenario_id,
            "ground_truth": ground_truth,
        },
        "env_kwargs": json.dumps(
            {"env_type": "telelogs", "case": case, "ground_truth": ground_truth}, ensure_ascii=False
        ),
    }


def _load_dataset(args: argparse.Namespace):
    import datasets

    token = os.environ.get("HF_TOKEN")
    split_files = {
        split: str(Path(value).expanduser())
        for split, value in (("train", args.local_train_file), ("test", args.local_test_file))
        if value
    }
    if args.local_dataset_path and split_files:
        raise ValueError("Use either --local-dataset-path or the split-specific local file options, not both")
    if split_files:
        return datasets.load_dataset("json", data_files=split_files)
    if args.local_dataset_path:
        path = Path(args.local_dataset_path).expanduser()
        if path.suffix in {".json", ".jsonl"}:
            return datasets.load_dataset("json", data_files=str(path))
        if path.suffix == ".parquet":
            return datasets.load_dataset("parquet", data_files=str(path))
        return datasets.load_from_disk(str(path))
    return datasets.load_dataset(args.dataset_name, token=token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="netop/TeleLogs")
    parser.add_argument("--local-dataset-path", default=None)
    parser.add_argument("--local-train-file", default=None)
    parser.add_argument("--local-test-file", default=None)
    parser.add_argument("--output-dir", default="~/data/telelogs_agent_r1")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    try:
        dataset = _load_dataset(args)
    except Exception as exc:
        raise SystemExit(
            "Unable to load TeleLogs. Accept the dataset conditions at "
            "https://huggingface.co/datasets/netop/TeleLogs and export HF_TOKEN, "
            "or pass --local-dataset-path/--local-train-file/--local-test-file. "
            f"Original error: {exc}"
        ) from exc

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_aliases = {"validation": "test", "val": "test"}
    for source_split in dataset:
        output_split = split_aliases.get(source_split, source_split)
        if output_split not in {"train", "test"}:
            continue
        split_dataset = dataset[source_split]
        limit = args.max_train_samples if output_split == "train" else args.max_test_samples
        if limit is not None:
            split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))

        def convert_current_split(record: dict[str, Any], idx: int, *, split_name: str = output_split):
            return convert_record(record, idx, split_name, args.dataset_name)

        converted = split_dataset.map(convert_current_split, with_indices=True)
        converted.to_parquet(output_dir / f"{output_split}.parquet")
        print(f"wrote {len(converted)} samples to {output_dir / f'{output_split}.parquet'}")


if __name__ == "__main__":
    main()
