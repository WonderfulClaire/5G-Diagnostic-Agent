import json

from recipes.telelogs.evaluation.harness_variants import (
    alias_tool_schemas,
    available_variants,
    canonicalize_tool_call_text,
)


SCHEMAS = [
    {"type": "function", "function": {"name": "query_radio_kpi", "parameters": {}}},
    {"type": "function", "function": {"name": "query_resource", "parameters": {}}},
    {"type": "function", "function": {"name": "submit_diagnosis", "parameters": {}}},
]


def _names(schemas):
    return [schema["function"]["name"] for schema in schemas]


def test_variants_change_query_schema_but_keep_submission_contract():
    canonical = alias_tool_schemas(SCHEMAS, "canonical")
    compact = alias_tool_schemas(SCHEMAS, "compact")
    alternate = alias_tool_schemas(SCHEMAS, "alternate")

    assert _names(canonical) == ["query_radio_kpi", "query_resource", "submit_diagnosis"]
    assert _names(compact) == ["radio", "resource", "submit_diagnosis"]
    assert _names(alternate) == [
        "inspect_radio_metrics",
        "inspect_resource_allocation",
        "submit_diagnosis",
    ]


def test_alias_call_is_canonicalized_before_environment_execution():
    raw = (
        "<tool_call>"
        + json.dumps({"name": "inspect_radio_metrics", "arguments": {"limit": 5}})
        + "</tool_call>"
    )
    canonical = canonicalize_tool_call_text(raw, "alternate")
    payload = json.loads(canonical.removeprefix("<tool_call>").removesuffix("</tool_call>"))
    assert payload["name"] == "query_radio_kpi"
    assert payload["arguments"] == {"limit": 5}


def test_unknown_or_malformed_calls_are_not_silently_rewritten():
    unknown = '<tool_call>{"name":"something_else","arguments":{}}</tool_call>'
    assert canonicalize_tool_call_text(unknown, "compact") == unknown

    malformed = "<tool_call>{bad json}</tool_call>"
    assert canonicalize_tool_call_text(malformed, "compact") == malformed


def test_variant_list_is_explicit_and_stable():
    assert available_variants() == ("canonical", "compact", "alternate")
