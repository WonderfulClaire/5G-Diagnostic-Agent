"""Record device/runtime versions before a training launch; fail closed."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cuda", "npu", "cpu"], default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/preflight.json"))
    a = p.parse_args()
    report = {
        "requested_backend": a.backend,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "packages": {},
    }
    for name in ["vllm", "transformers", "peft", "ray", "torch-npu", "vllm-ascend", "verl"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
    report["training_ready"] = (
        a.backend == "cuda"
        and torch.cuda.is_available()
        and all(report["packages"][x] for x in ["vllm", "transformers", "peft", "ray", "verl"])
    )
    if a.backend == "npu":
        report["limitation"] = (
            "NPU trainer/rollout port has not been validated. Do not infer support from package presence."
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["training_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
