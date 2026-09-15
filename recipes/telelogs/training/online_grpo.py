"""Transparent single-GPU LoRA-GRPO over real multi-turn model rollouts.

Each response is rescored with its exact observed prefix. Tool observations never
enter the target mask. This reference trainer prioritizes auditability over throughput.
"""

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import torch
from agent_r1.env.base import Action
from recipes.telelogs.env.telelogs_env import TeleLogsEnv
from recipes.telelogs.prompts import build_agent_messages
from .objective import group_advantages, action_loss, learning_route


def encode_prompt(tokenizer, messages, schemas, device):
    text = tokenizer.apply_chat_template(
        messages, tools=schemas, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def log_probs(model, prefix, actions):
    ids = torch.cat((prefix, actions), dim=1)
    logits = model(ids[:, :-1]).logits[:, prefix.shape[1] - 1 :, :].float()
    return logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(0).squeeze(-1)


async def sample_episode(model, tokenizer, case, device, max_steps, max_tokens):
    env = TeleLogsEnv(case=case["case"], ground_truth=case["ground_truth"])
    messages = build_agent_messages(case["id"], case["symptom"])
    env.reset(raw_prompt=messages)
    pieces = []
    trace = []
    total = 0.0
    submitted = False
    for step in range(max_steps):
        prefix = encode_prompt(tokenizer, messages, env.tool_schemas, device)
        if prefix.shape[1] + max_tokens > 8192:
            break
        with torch.no_grad():
            full = model.generate(
                prefix,
                attention_mask=torch.ones_like(prefix),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        actions = full[:, prefix.shape[1] :]
        text = tokenizer.decode(actions[0], skip_special_tokens=True)
        observation, reward, done, info = await env.step(Action(text=text))
        messages = list(observation.messages)
        pieces.append({"prefix": prefix, "actions": actions})
        trace.append(
            {
                "step": step,
                "action": text,
                "reward": reward,
                "info": info,
                "context_tokens": prefix.shape[1],
                "action_tokens": actions.shape[1],
            }
        )
        total += reward
        submitted |= any("predicted_root_causes" in c for c in info.get("tool_calls", []))
        if done:
            break
    if not submitted:
        total -= 0.2
    return pieces, total, trace


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rows = [json.loads(x) for x in args.cases.read_text().splitlines() if x.strip()]
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("Require train-only cases")
    if args.group_size < 2:
        raise ValueError("GRPO group size must be >=2")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(
        args.device
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
        if args.adapter
        else get_peft_model(
            base,
            LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
            ),
        )
    )
    model.eval()
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "seed": args.seed,
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "group_size": args.group_size,
        "steps": args.steps,
        "learning_rate": args.lr,
        "beta": args.beta,
        "torch": torch.__version__,
        "scope": "synthetic_or_user_supplied_task; inspect dataset provenance",
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
    for step in range(args.steps):
        case = rows[step % len(rows)]
        episodes = []
        rewards = []
        correctness = []
        efficiency_costs = []
        for member in range(args.group_size):
            pieces, reward, trace = asyncio.run(
                sample_episode(model, tokenizer, case, args.device, args.max_steps, args.max_tokens)
            )
            efficiency_costs.append(sum(len(t.get("info", {}).get("tool_calls", [])) for t in trace))
            finals = [c for t in trace for c in t.get("info", {}).get("tool_calls", []) if "predicted_root_causes" in c]
            pred = set(finals[-1]["predicted_root_causes"]) if finals else set()
            truth = set(case["ground_truth"])
            correctness.append(2 * len(pred & truth) / (len(pred) + len(truth)))
            episodes.append(pieces)
            rewards.append(reward)
            with (args.output / "trajectories.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        {
                            "step": step,
                            "member": member,
                            "case_id": case["id"],
                            "split": "train",
                            "reward": reward,
                            "trace": trace,
                        }
                    )
                    + "\n"
                )
        advantages, std = group_advantages(rewards)
        route = learning_route(rewards, correctness, efficiency_costs=efficiency_costs)
        metrics = {
            "learning_route": route,
            "optimizer_updated": False,
            "correctness_scores": correctness,
            "efficiency_costs": efficiency_costs,
            "step": step,
            "case_id": case["id"],
            "rewards": rewards,
            "group_reward_std": std,
            "skipped_equal_reward": std < 1e-6,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        if route in {"rl_ready", "efficiency_rl"}:
            for pieces in episodes:
                for piece in pieces:
                    with torch.no_grad():
                        piece["old"] = log_probs(model, piece["prefix"], piece["actions"]).detach()
                        piece["ref"] = log_probs(reference, piece["prefix"], piece["actions"]).detach()
            optimizer.zero_grad()
            kl = 0.0
            clipped = 0
            tokens = 0
            # Each trajectory has equal weight, regardless of how many calls it made.
            for index, pieces in enumerate(episodes):
                count = sum(p["actions"].numel() for p in pieces)
                for piece in pieces:
                    new = log_probs(model, piece["prefix"], piece["actions"])
                    loss, parts = action_loss(
                        new, piece["old"], piece["ref"], advantages[index].to(args.device), beta=args.beta
                    )
                    (loss / (max(1, count) * args.group_size)).backward()
                    kl += parts["kl_sum"]
                    clipped += parts["clip_count"]
                    tokens += parts["tokens"]
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError("Nonfinite gradient; stop before optimizer update")
            optimizer.step()
            metrics["optimizer_updated"] = True
            metrics.update(kl=kl / max(1, tokens), clip_fraction=clipped / max(1, tokens), grad_norm=float(grad))
        with (args.output / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    torch.save(optimizer.state_dict(), args.output / "optimizer.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cases", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.01)
    train(p.parse_args())


if __name__ == "__main__":
    main()
