"""Bounded tool-use harness with auditable trajectories and an OpenAI-compatible backend.

API requests expose only the public prompt and observations, never case state or labels.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from agent_r1.env.base import Action
from recipes.telelogs.env.telelogs_env import TeleLogsEnv
from recipes.telelogs.prompts import build_agent_messages
from recipes.telelogs.evaluation.evaluate_predictions import evaluate_rows


class ChatBackend:
    def __init__(self, base_url, model):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model

    def __call__(self, messages, schemas):
        payload = {"model": self.model, "messages": messages, "tools": schemas, "temperature": 0, "max_tokens": 1024}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("MODEL_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        with urlopen(Request(self.url, data=json.dumps(payload).encode(), headers=headers), timeout=180) as r:
            msg = json.load(r)["choices"][0]["message"]
        calls = msg.get("tool_calls", [])
        if calls:
            return "".join(
                "<tool_call>"
                + json.dumps({"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])})
                + "</tool_call>"
                for c in calls
            )
        return msg.get("content") or ""


async def run_case(case, backend, max_steps=8):
    env = TeleLogsEnv(case=case["case"], ground_truth=case["ground_truth"])
    messages = build_agent_messages(case["id"], case.get("symptom", "Downlink throughput is below 600 Mbps."))
    env.reset(raw_prompt=messages)
    trace = []
    done = False
    submitted = None
    failures = 0
    count = 0
    for step in range(max_steps):
        text = await asyncio.to_thread(backend, messages, env.tool_schemas)
        obs, reward, done, info = await env.step(Action(text=text))
        calls = info.get("tool_calls", [])
        count += len(calls)
        failures += sum(bool(c.get("error")) for c in calls)
        trace.append({"step": step, "action": text, "observation": obs.messages, "reward": reward, "info": info})
        # The environment already returns the complete conversation. Replacing it
        # prevents recursively embedding all earlier messages in each observation.
        messages = list(obs.messages)
        for call in calls:
            if "predicted_root_causes" in call:
                submitted = call
        if done:
            break
    predicted = submitted.get("predicted_root_causes", []) if submitted else []
    error = []
    if failures:
        error.append("tool_failure")
    if not submitted:
        error.append("missing_submission" if done else "step_budget_exhausted")
    if submitted and set(predicted) != set(case["ground_truth"]):
        error.append("root_cause_error")
    return {
        "scenario_id": case["id"],
        "split": case.get("split", "unknown"),
        "ground_truth": case["ground_truth"],
        "predicted_root_causes": predicted,
        "tool_call_count": count,
        "tool_failure_count": failures,
        "iterations": len(trace),
        "error_categories": error,
        "trace": trace,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cases", type=Path)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--local", action="store_true", help="Interpret --model as a local Transformers checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=8)
    a = p.parse_args()
    if a.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if a.local:
        from .local_model import LocalModelBackend

        backend = LocalModelBackend(a.model, device=a.device, adapter=a.adapter)
    else:
        backend = ChatBackend(a.base_url, a.model)
    rows = [
        asyncio.run(run_case(json.loads(line), backend, a.max_steps))
        for line in a.cases.read_text().splitlines()
        if line.strip()
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    metrics = evaluate_rows(rows)
    a.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
