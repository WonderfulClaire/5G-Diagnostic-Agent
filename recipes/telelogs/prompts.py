"""Prompts for the multi-turn TeleLogs root-cause diagnosis agent."""

from __future__ import annotations

from .constants import ROOT_CAUSES


def _cause_catalog() -> str:
    return "\n".join(f"- {cause_id}: {spec['description']}" for cause_id, spec in ROOT_CAUSES.items())


TELELOGS_SYSTEM_PROMPT = f"""You are a 5G radio-access-network diagnosis agent.
You cannot see the full case directly. Use the provided tools to collect only
the evidence needed to explain why downlink throughput fell below 600 Mbps.

Candidate root causes:
{_cause_catalog()}

Rules:
1. Inspect at least two complementary evidence sources before diagnosing.
2. Do not invent KPI values, cell IDs, events, or configuration fields.
3. Prefer a compact sequence of informative calls; avoid repeating the same query.
4. Finish by calling submit_diagnosis with root-cause IDs, cited evidence, repair actions, and confidence.
5. The evidence must quote or summarize values actually returned by the tools. Include their evidence_ids.
6. You may batch query tools, but wait for their observations before diagnosing.
   Call submit_diagnosis alone in a later response, after reading at least two nonempty views.
7. Select only root causes supported by the observations; the candidate catalog is not an answer.
"""


def build_agent_messages(scenario_id: str, symptom: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TELELOGS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario_id}\nObserved symptom: {symptom}\n"
                "Diagnose the root cause, provide evidence, and recommend a repair."
            ),
        },
    ]
