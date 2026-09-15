"""Decision-focused SFT/DPO with verified synthetic near-miss candidates.

The environment executes teacher queries; hidden labels are target data only.
The frozen initial model ranks rule-constructed negatives. Full free-generation
tool evaluation is required separately; pair accuracy is not Agent success.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .online_grpo import encode_prompt, log_probs
from .supervised import demonstrations


def preference_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (chosen - rejected) - (ref_chosen.detach() - ref_rejected.detach())
    return -F.logsigmoid(beta * margin)


def decision_tokens(tokenizer, header, alternatives):
    # Tokenize complete strings and use a common boundary so token merges at
    # the header/list boundary cannot silently change the compared prefix.
    encoded = [tokenizer(header + json.dumps(value) + ",", add_special_tokens=False).input_ids
               for value in alternatives]
    boundary = max(0, len(tokenizer(header, add_special_tokens=False).input_ids) - 1)
    while boundary and any(x[:boundary] != encoded[0][:boundary] for x in encoded[1:]):
        boundary -= 1
    return encoded[0][:boundary], [x[boundary:] for x in encoded]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    p = argparse.ArgumentParser()
    p.add_argument("cases", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dpo-weight", type=float, default=1)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.steps < 1 or a.lr <= 0 or a.beta <= 0 or a.dpo_weight < 0:
        raise ValueError("Invalid training configuration")
    rows = [json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()]
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("Need training-only preferences")
    for row in rows:
        if not row.get("preference_candidates") or any(
            not c or set(c) == set(row["ground_truth"]) for c in row["preference_candidates"]
        ):
            raise ValueError("Invalid preference candidates")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(a.device)
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True).eval()
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    samples = []
    mined = []

    async def collect():
        for row in rows:
            async for messages, response, schemas in demonstrations(row):
                match = re.search(r'"root_causes"\s*:\s*\[', response)
                if match is None:
                    continue
                header = response[:match.end()-1]
                options = [sorted(row["ground_truth"])] + row["preference_candidates"]
                extra, candidates = decision_tokens(tokenizer, header, options)
                prefix = torch.cat((encode_prompt(tokenizer, messages, schemas, a.device),
                                    torch.tensor([extra], device=a.device, dtype=torch.long)), 1)
                targets = [torch.tensor([x], device=a.device) for x in candidates]
                with torch.no_grad():
                    ref = [log_probs(reference, prefix, target).sum() for target in targets]
                hardest = max(range(1, len(ref)), key=lambda i: float(ref[i]))
                samples.append((prefix, targets[0], targets[hardest], ref[0], ref[hardest],
                                len(row["ground_truth"]) > 1))
                mined.append({"id": row["id"], "split": "train", "chosen": options[0],
                              "rejected": options[hardest], "reference_scores": [float(x) for x in ref],
                              "audit": row.get("preference_audit", {})})
            print(json.dumps({"mined_cases": len(samples)}), flush=True)

    asyncio.run(collect())
    (a.output / "mined.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mined))
    random.Random(a.seed).shuffle(samples)
    old = [x for x in samples if not x[-1]]
    new = [x for x in samples if x[-1]]
    config = {"seed": a.seed, "steps": a.steps, "lr": a.lr, "beta": a.beta,
              "dpo_weight": a.dpo_weight, "sft_weight": 1,
              "data_sha256": hashlib.sha256(a.cases.read_bytes()).hexdigest(),
              "schedule": "4 single-cause decisions to 1 compound decision",
              "samples": len(samples), "scope": "synthetic decision learning; full tool generation evaluated separately"}
    (a.output / "config.json").write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr)
    for step in range(a.steps):
        sample = (new[(step//5) % len(new)] if step % 5 == 4 else old[(step-step//5) % len(old)]) if old and new else samples[step % len(samples)]
        prefix, chosen, rejected, ref_chosen, ref_rejected, is_new = sample
        cp = log_probs(model, prefix, chosen)
        ce = -cp.mean()
        dpo = torch.zeros((), device=a.device)
        if a.dpo_weight:
            rp = log_probs(model, prefix, rejected)
            dpo = preference_loss(cp.sum(), rp.sum(), ref_chosen, ref_rejected, a.beta)
        loss = ce + a.dpo_weight * dpo
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        result = {"step": step, "loss": float(loss.detach()), "ce": float(ce.detach()),
                  "dpo": float(dpo.detach()), "gradient_norm": float(norm), "is_new": is_new,
                  "chosen_tokens": chosen.numel(), "rejected_tokens": rejected.numel() if a.dpo_weight else 0}
        with (a.output / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(result) + "\n")
        if step % 10 == 0:
            print(json.dumps(result), flush=True)
    model.save_pretrained(a.output / "adapter")
    tokenizer.save_pretrained(a.output / "adapter")


if __name__ == "__main__":
    main()
