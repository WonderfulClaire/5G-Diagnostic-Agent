"""LoRA cold start from executable training-only tool trajectories."""

import copy
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import torch
from agent_r1.env.base import Action
from recipes.telelogs.env.telelogs_env import TeleLogsEnv
from recipes.telelogs.constants import recommended_repairs
from recipes.telelogs.prompts import build_agent_messages
from .online_grpo import encode_prompt, log_probs


def tool_call(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


async def demonstrations(case):
    env = TeleLogsEnv(case=case["case"], ground_truth=case["ground_truth"])
    messages = build_agent_messages(case["id"], case["symptom"])
    env.reset(raw_prompt=messages)
    # Demonstration queries all views. It does not pretend to be a learned efficient policy.
    query = (
        tool_call("query_radio_kpi", {})
        + tool_call("query_cell_relation", {})
        + tool_call("query_mobility", {})
        + tool_call("query_antenna", {})
        + tool_call("query_handover", {})
        + tool_call("query_resource", {})
    )
    yield list(messages), query, env.tool_schemas
    observation, _, done, info = await env.step(Action(text=query))
    if done or any(c.get("error") for c in info.get("tool_calls", [])):
        raise ValueError("Invalid demonstration query")
    messages = list(observation.messages)
    causes = case["ground_truth"]
    cause_views = {
        "C1": "mobility",
        "C2": "antenna",
        "C3": "cell_relation",
        "C4": "cell_relation",
        "C5": "cell_relation",
        "C6": "handover",
        "C7": "handover",
        "C8": "resource",
    }
    entries = [e for e in env.evidence_records.values() if e["view"] in {cause_views[c] for c in causes}]
    args = {
        "root_causes": causes,
        "evidence": [str(e["record"]) for e in entries],
        "evidence_ids": [e["evidence_id"] for e in entries],
        "repair_actions": recommended_repairs(causes),
        "confidence": 0.9,
    }
    submit = tool_call("submit_diagnosis", args)
    yield messages, submit, env.tool_schemas
    _, _, done, info = await env.step(Action(text=submit))
    if not done or info["tool_calls"][0].get("predicted_root_causes") != causes:
        raise ValueError("Invalid teacher submission")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    p = argparse.ArgumentParser()
    p.add_argument("cases", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--replay-kl", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    rows = [json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()]
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("Need training cases only")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(
        a.device
    )
    model = (
        PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
        if a.adapter
        else get_peft_model(
            base,
            LoraConfig(r=16, lora_alpha=32, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
    )
    model.eval()
    reference = copy.deepcopy(model).requires_grad_(False).eval() if a.replay_kl > 0 else None
    samples = []

    async def collect():
        for case in rows:
            async for messages, response, schemas in demonstrations(case):
                samples.append(
                    (
                        encode_prompt(tokenizer, messages, schemas, a.device),
                        tokenizer(
                            response + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False
                        ).input_ids.to(a.device),
                        case.get("curation", {}).get("status") == "accept",
                    )
                )

    asyncio.run(collect())
    rng.shuffle(samples)
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr)
    replay = [x for x in samples if not x[2]]
    fresh = [x for x in samples if x[2]]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "config.json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "init_adapter": a.adapter,
                "lr": a.lr,
                "replay_kl": a.replay_kl,
                "schedule": "four replay steps per new-data step" if replay and fresh else "uniform",
                "steps": a.steps,
                "examples": len(samples),
                "data_sha256": hashlib.sha256(a.cases.read_bytes()).hexdigest(),
                "scope": "synthetic executable demonstration cold start",
            },
            indent=2,
        )
    )
    for step in range(a.steps):
        if replay and fresh:
            sample = fresh[(step // 5) % len(fresh)] if step % 5 == 4 else replay[(step - step // 5) % len(replay)]
        else:
            sample = samples[step % len(samples)]
        prefix, actions, is_new = sample
        loss = -log_probs(model, prefix, actions).mean()
        kl = torch.tensor(0.0, device=a.device)
        if reference is not None and not is_new:
            ids = torch.cat((prefix, actions), 1)
            logits = model(ids[:, :-1]).logits[:, prefix.shape[1] - 1 :, :].float()
            with torch.no_grad():
                ref_logits = reference(ids[:, :-1]).logits[:, prefix.shape[1] - 1 :, :].float()
            logq = logits.log_softmax(-1)
            logp = ref_logits.log_softmax(-1)
            kl = (logp.exp() * (logp - logq)).sum(-1).mean()
            loss = loss + a.replay_kl * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "replay_kl": float(kl.detach()), "is_new": is_new}
        with (a.output / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
    model.save_pretrained(a.output / "adapter")
    tokenizer.save_pretrained(a.output / "adapter")


if __name__ == "__main__":
    main()
