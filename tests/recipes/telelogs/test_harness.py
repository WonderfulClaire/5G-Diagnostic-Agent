import asyncio
import json
from recipes.telelogs.evaluation.run_agent import run_case
from tests.recipes.telelogs.test_environment import tool_call


def test_scenario_identifiers_never_enter_model_context():
    from recipes.telelogs.prompts import build_agent_messages
    messages = build_agent_messages("train-answer-C1-C8-sensitive-id", "Slow downlink")
    assert "train-answer-C1-C8-sensitive-id" not in json.dumps(messages)
    assert "Slow downlink" in messages[1]["content"]


def test_runner_keeps_hidden_case_out_of_prompt_and_records_success():
    answers = iter(
        [
            tool_call("query_radio_kpi", {}),
            tool_call("query_cell_relation", {}),
            tool_call(
                "submit_diagnosis",
                {
                    "root_causes": ["C4"],
                    "evidence": ["SINR -5 dB and co-frequency neighbor"],
                    "repair_actions": ["interference coordination"],
                    "confidence": 0.9,
                },
            ),
        ]
    )
    history = []

    def backend(messages, schemas):
        history.append(json.loads(json.dumps(messages)))
        return next(answers)

    case = {
        "id": "fixture",
        "split": "synthetic",
        "ground_truth": ["C4"],
        "case": {"sections": {"radio_kpi": ["SINR -5 dB"], "cell_relation": ["non-colocated co-frequency neighbor"]}},
    }
    row = asyncio.run(run_case(case, backend))
    assert row["predicted_root_causes"] == ["C4"] and row["tool_call_count"] == 3
    assert not row["error_categories"]
    assert "SINR -5 dB" not in json.dumps(history[0])
    assert "SINR -5 dB" in json.dumps(history[1])
    assert all("ground_truth" not in json.dumps(h) for h in history)


def test_runner_has_step_budget():
    case = {"id": "fixture", "ground_truth": ["C4"], "case": {"sections": {"radio_kpi": ["SINR -5 dB"]}}}
    row = asyncio.run(run_case(case, lambda messages, schemas: tool_call("query_radio_kpi", {}), max_steps=2))
    assert row["iterations"] == 2 and "step_budget_exhausted" in row["error_categories"]


def test_context_does_not_recursively_duplicate_history():
    snapshots = []

    def backend(messages, schemas):
        snapshots.append(list(messages))
        return tool_call("query_radio_kpi", {})

    case = {"id": "fixture", "ground_truth": ["C4"], "case": {"sections": {"radio_kpi": ["SINR -5 dB"]}}}
    asyncio.run(run_case(case, backend, max_steps=4))
    assert [len(x) for x in snapshots] == [2, 4, 6, 8]
    assert all(sum(m["role"] == "system" for m in x) == 1 for x in snapshots)
