"""A hidden-state tool environment for TeleLogs RCA.

The original TeleLogs rows expose the complete engineering context.  This
environment hides that context behind six diagnostic views so Agent-R1 learns
an actual multi-step observation/action/reward process instead of single-turn
classification.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections import defaultdict
from typing import Any

from agent_r1.env.base import Action, AgentEnv, Observation
from agent_r1.env.tool_format import ToolFormatWrapper

from ..constants import ROOT_CAUSES, normalize_root_causes, recommended_repairs, set_f1

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "radio_kpi": (
        "throughput",
        "rsrp",
        "sinr",
        "timestamp",
        "serving cell",
        "user-plane",
    ),
    "cell_relation": (
        "pci",
        "neighbor",
        "neighbour",
        "cell id",
        "gnodeb",
        "co-frequency",
        "cochannel",
    ),
    "mobility": (
        "latitude",
        "longitude",
        "location",
        "speed",
        "distance",
        "trajectory",
    ),
    "antenna": ("azimuth", "downtilt", "tilt", "antenna", "height", "beam"),
    "handover": (
        "handover",
        "signaling",
        "event a3",
        "threshold",
        "hysteresis",
        "time-to-trigger",
        "ttt",
    ),
    "resource": (
        "resource block",
        "scheduled rb",
        "allocated rb",
        "rb num",
        "rbs",
        "prb",
        "scheduler",
    ),
}

SECTION_COLUMN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "radio_kpi": (
        "timestamp",
        "serving pci",
        "serving ss-rsrp",
        "serving ss-sinr",
        "throughput",
    ),
    "cell_relation": ("timestamp", "gnodeb", "cell id", "pci"),
    "mobility": ("timestamp", "longitude", "latitude", "speed", "distance"),
    "antenna": (
        "gnodeb",
        "cell id",
        "longitude",
        "latitude",
        "azimuth",
        "downtilt",
        "digital tilt",
        "beam scenario",
        "height",
        "pci",
    ),
    "handover": (
        "timestamp",
        "handover",
        "signaling",
        "event a3",
        "threshold",
        "hysteresis",
        "ttt",
    ),
    "resource": (
        "timestamp",
        "resource block",
        "scheduled rb",
        "allocated rb",
        "rb num",
        "rbs",
        "prb",
    ),
}

SECTION_REQUIRED_COLUMN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "radio_kpi": ("serving ss-rsrp", "serving ss-sinr", "throughput"),
    "cell_relation": ("gnodeb", "cell id", "pci"),
    "mobility": ("longitude", "latitude", "speed", "distance"),
    "antenna": ("azimuth", "downtilt", "digital tilt", "beam scenario", "height"),
    "handover": ("handover", "signaling", "event a3", "threshold", "hysteresis", "ttt"),
    "resource": (
        "resource block",
        "scheduled rb",
        "allocated rb",
        "rb num",
        "rbs",
        "prb",
    ),
}

QUERY_TO_SECTION = {
    "query_radio_kpi": "radio_kpi",
    "query_cell_relation": "cell_relation",
    "query_mobility": "mobility",
    "query_antenna": "antenna",
    "query_handover": "handover",
    "query_resource": "resource",
}


def _parse_number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text in {"-", "--", "N/A", "null", "None"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _parse_pipe_table(block: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip().strip("|") for line in block.splitlines() if "|" in line]
    if len(lines) < 2:
        return None
    headers = [part.strip() for part in lines[0].split("|")]
    if len(headers) < 2:
        return None
    rows: list[list[str]] = []
    for line in lines[1:]:
        values = [part.strip() for part in line.split("|")]
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        rows.append(values[: len(headers)])
    return headers, rows


def _summarize_columns(headers: list[str], rows: list[list[str]], indices: list[int]) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for index in indices:
        header = headers[index]
        values = [row[index] for row in rows]
        present = [value for value in values if value not in {"", "-", "--", "N/A"}]
        unique = list(dict.fromkeys(present))
        numeric = [number for value in present if (number := _parse_number(value)) is not None]
        summary: dict[str, Any] = {
            "count": len(present),
            "missing": len(values) - len(present),
            "transitions": sum(left != right for left, right in zip(present, present[1:], strict=False)),
        }
        mostly_numeric = bool(numeric) and len(numeric) >= max(1, int(0.8 * len(present)))
        if mostly_numeric:
            summary.update(
                {
                    "min": round(min(numeric), 6),
                    "max": round(max(numeric), 6),
                    "mean": round(sum(numeric) / len(numeric), 6),
                }
            )
        if not mostly_numeric and len(unique) <= 8:
            summary["unique"] = unique
        elif not mostly_numeric and unique:
            summary["first"] = unique[0]
            summary["last"] = unique[-1]
        statistics[header] = summary

    sample_indices = [0] if rows else []
    if len(rows) > 1:
        sample_indices.append(len(rows) - 1)
    sample_rows = [
        {headers[index]: rows[row_index][index] for index in indices} for row_index in dict.fromkeys(sample_indices)
    ]
    return {
        "source": "pipe_table_summary",
        "row_count": len(rows),
        "columns": [headers[index] for index in indices],
        "statistics": statistics,
        "sample_rows": sample_rows,
    }


def split_case_document(document: str) -> dict[str, list[Any]]:
    """Build compact, deterministic tool views from a raw TeleLogs document.

    Pipe-delimited tables are projected onto section-specific columns and
    summarized. This preserves sequence changes and numeric extrema without
    returning the same full table through every tool.
    """

    normalized = str(document).replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized) if block.strip()]
    if len(blocks) <= 1 and normalized:
        blocks = [segment.strip() for segment in re.split(r"(?<=[.;])\s+", normalized) if segment.strip()]

    sections: dict[str, list[Any]] = defaultdict(list)
    for block in blocks:
        parsed_table = _parse_pipe_table(block)
        if parsed_table is not None:
            headers, rows = parsed_table
            for section, keywords in SECTION_COLUMN_KEYWORDS.items():
                required_keywords = SECTION_REQUIRED_COLUMN_KEYWORDS[section]
                if not any(required in header.casefold() for header in headers for required in required_keywords):
                    continue
                indices = [
                    index
                    for index, header in enumerate(headers)
                    if any(keyword in header.casefold() for keyword in keywords)
                ]
                if indices:
                    sections[section].append(_summarize_columns(headers, rows, indices))
            continue

        collapsed = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in block.splitlines())
        lowered = collapsed.casefold()
        for section, keywords in SECTION_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                sections[section].append(collapsed)

    return {section: sections.get(section, []) for section in SECTION_KEYWORDS}


def extract_symptom(document: str) -> str:
    for line in str(document).splitlines():
        lowered = line.casefold()
        if "symptom" in lowered or ("throughput" in lowered and ("600" in lowered or "degrad" in lowered)):
            return re.sub(r"\s+", " ", line).strip()[:500]
    return "Downlink throughput degraded below 600 Mbps during the drive test."


def _tool_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Optional keyword used to filter the returned records.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum records to return; defaults to 20 and is capped at 50.",
                    },
                },
                "required": [],
            },
        },
    }


TELELOGS_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool_schema(
        "query_radio_kpi",
        "Query throughput, RSRP, SINR, serving-cell, and timestamp records.",
    ),
    _tool_schema(
        "query_cell_relation",
        "Query serving/neighbor cell IDs, PCIs, and co-frequency relations.",
    ),
    _tool_schema("query_mobility", "Query UE position, speed, distance, and trajectory records."),
    _tool_schema(
        "query_antenna",
        "Query antenna azimuth, downtilt, height, and beam configuration.",
    ),
    _tool_schema(
        "query_handover",
        "Query signaling events, handovers, thresholds, hysteresis, and TTT.",
    ),
    _tool_schema(
        "query_resource",
        "Query scheduled resource-block and scheduler-allocation records.",
    ),
    {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": "Submit the final root cause, evidence, repair actions, and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root_causes": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ROOT_CAUSES)},
                        "description": "One or more IDs from C1 through C8.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Evidence grounded in tool observations.",
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs returned by prior queries; cite the records supporting the diagnosis.",
                    },
                    "repair_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete configuration or operational remediation steps.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Calibrated confidence from 0 to 1.",
                    },
                },
                "required": ["root_causes", "evidence", "repair_actions", "confidence"],
            },
        },
    },
]


def _content_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) >= 3}


def _grounded_evidence_score(evidence: list[str], observed_text: str) -> float:
    if not evidence:
        return 0.0
    observed_tokens = _content_tokens(observed_text)
    grounded = 0
    for item in evidence:
        tokens = _content_tokens(str(item))
        if tokens and len(tokens & observed_tokens) / len(tokens) >= 0.35:
            grounded += 1
    return grounded / len(evidence)


def _cause_evidence_score(predicted: list[str], evidence: list[str]) -> float:
    if not predicted:
        return 0.0
    evidence_text = " ".join(map(str, evidence)).casefold()
    supported = 0
    for cause_id in predicted:
        keywords = ROOT_CAUSES[cause_id]["evidence_keywords"]
        if any(keyword.casefold() in evidence_text for keyword in keywords):
            supported += 1
    return supported / len(predicted)


def _repair_score(predicted: list[str], repair_actions: list[str]) -> float:
    if not predicted:
        return 0.0
    repair_text = " ".join(map(str, repair_actions)).casefold()
    covered = 0
    for cause_id in predicted:
        keywords = ROOT_CAUSES[cause_id]["repair_keywords"]
        if any(keyword.casefold() in repair_text for keyword in keywords):
            covered += 1
    return covered / len(predicted)


def score_diagnosis(
    *,
    predicted: list[str],
    expected: list[str],
    evidence: list[str],
    repair_actions: list[str],
    observed_text: str,
    query_count: int,
) -> tuple[float, dict[str, float]]:
    """Score correctness, grounded evidence, remediation, and tool efficiency."""

    root_cause_f1 = set_f1(predicted, expected)
    grounded = _grounded_evidence_score(evidence, observed_text)
    causal_support = _cause_evidence_score(predicted, evidence)
    evidence_score = 0.5 * grounded + 0.5 * causal_support
    repair_score = _repair_score(predicted, repair_actions)
    efficiency = max(0.0, 1.0 - 0.1 * max(0, query_count - 4)) if query_count >= 2 else 0.25 * query_count
    total = 0.70 * root_cause_f1 + 0.15 * evidence_score + 0.10 * repair_score + 0.05 * efficiency
    components = {
        "root_cause_f1": root_cause_f1,
        "evidence_groundedness": grounded,
        "cause_evidence_support": causal_support,
        "repair_relevance": repair_score,
        "tool_efficiency": efficiency,
    }
    return round(min(1.0, max(0.0, total)), 6), components


@AgentEnv.register("telelogs")
class TeleLogsEnv(AgentEnv):
    """Per-sample TeleLogs diagnostic environment for Agent-R1 rollouts."""

    def __init__(
        self,
        case: dict[str, Any],
        ground_truth: list[str] | str,
        tool_format: str = "hermes",
        new_evidence_reward: float = 0.02,
        repeat_tool_penalty: float = -0.05,
        invalid_tool_penalty: float = -0.20,
        missing_submission_penalty: float = -0.50,
        query_cost: float = 0.005,
        **_: Any,
    ):
        self.query_cost = float(query_cost)
        if self.query_cost < 0:
            raise ValueError("query_cost must be nonnegative")
        self.case = dict(case)
        self.ground_truth = normalize_root_causes(ground_truth)
        if not self.ground_truth:
            raise ValueError("TeleLogsEnv requires at least one normalized ground-truth cause")
        self.format_wrapper = ToolFormatWrapper.from_name(tool_format)
        self.new_evidence_reward = float(new_evidence_reward)
        self.repeat_tool_penalty = float(repeat_tool_penalty)
        self.invalid_tool_penalty = float(invalid_tool_penalty)
        self.missing_submission_penalty = float(missing_submission_penalty)
        sections = self.case.get("sections")
        self.sections = (
            sections if isinstance(sections, dict) else split_case_document(self.case.get("case_document", ""))
        )
        self._messages: list[dict[str, Any]] = []
        self._queried: list[str] = []
        self._observations: list[str] = []
        self._observed_views: set[str] = set()
        self.last_info: dict[str, Any] = {}
        self.evidence_records: dict[str, dict] = {}
        self._seen_records: set[str] = set()
        self._done = False

    def reset(self, **kwargs: Any) -> Observation:
        self._messages = list(kwargs.get("raw_prompt", []))
        self._queried = []
        self._observations = []
        self._observed_views = set()
        self.last_info = {}
        self.evidence_records = {}
        self._seen_records = set()
        self._done = False
        return Observation(messages=list(self._messages))

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return TELELOGS_TOOL_SCHEMAS

    def _query(self, name: str, args: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
        section = QUERY_TO_SECTION[name]
        records = list(self.sections.get(section, []))
        focus = str(args.get("focus", "")).strip().casefold()
        if focus:
            records = [record for record in records if focus in json.dumps(record, ensure_ascii=False).casefold()]
        try:
            limit = min(50, max(1, int(args.get("limit", 20))))
        except (TypeError, ValueError):
            limit = 20
        records = records[:limit]
        entries = []
        new_count = 0
        for record in records:
            canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
            # Novelty is content-based; returning the same text via another view earns no bonus.
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            evidence_id = section + ":" + fingerprint[:12]
            entry = {"evidence_id": evidence_id, "view": section, "record": record}
            entries.append(entry)
            new_count += int(fingerprint not in self._seen_records)
            self._seen_records.add(fingerprint)
            self.evidence_records[evidence_id] = entry
        payload = {"view": section, "record_count": len(records), "records": records, "evidence_records": entries}
        response = json.dumps(payload, ensure_ascii=False)
        repeated = bool(records) and new_count == 0
        self._queried.append(name)
        self._observations.append(response)
        if records:
            self._observed_views.add(section)
        bonus = self.new_evidence_reward if new_count else (self.repeat_tool_penalty if repeated else 0.0)
        reward = bonus - self.query_cost
        return (
            response,
            reward,
            {
                "tool": name,
                "repeated": repeated,
                "record_count": len(records),
                "new_record_count": new_count,
                "query_cost": self.query_cost,
                "evidence_ids": [entry["evidence_id"] for entry in entries],
            },
        )

    def _submit(self, args: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
        predicted = normalize_root_causes(args.get("root_causes"))
        evidence = [str(item) for item in args.get("evidence", [])] if isinstance(args.get("evidence"), list) else []
        repair_actions = (
            [str(item) for item in args.get("repair_actions", [])]
            if isinstance(args.get("repair_actions"), list)
            else []
        )
        try:
            confidence = min(1.0, max(0.0, float(args.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        score, components = score_diagnosis(
            predicted=predicted,
            expected=self.ground_truth,
            evidence=evidence,
            repair_actions=repair_actions,
            observed_text="\n".join(self._observations),
            query_count=len(self._queried),
        )
        response = {
            "accepted": bool(predicted),
            "score": score,
            "predicted_root_causes": predicted,
            "evidence": evidence,
            "repair_actions": repair_actions,
            "confidence": confidence,
            "evidence_ids": args.get("evidence_ids", []),
            "fallback_repair_actions": recommended_repairs(predicted) if not repair_actions else [],
        }
        info = {
            **response,
            "ground_truth": self.ground_truth,
            "score_components": components,
        }
        return json.dumps(response, ensure_ascii=False), score, info

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if not isinstance(action, Action) or action.text is None:
            raise TypeError("TeleLogsEnv accepts Action(text=...) only")

        if self._done:
            raise RuntimeError("Episode ended; call reset before another step")
        _, calls = self.format_wrapper.parse_response(action.text)
        self._messages.append({"role": "assistant", "content": action.text})
        opening = action.text.count("<tool_call>")
        closing = action.text.count("</tool_call>")
        if (opening or closing) and (opening != closing or len(calls) != opening):
            error = "malformed_tool_call"
            message = json.dumps(
                {
                    "error": error,
                    "hint": "Return complete tool_call blocks with valid JSON. No partial batch was executed.",
                }
            )
            self._messages.append({"role": "user", "content": self.format_wrapper.format_observation(message)})
            info = {"tool_calls": [{"tool": "parse", "error": error}], "query_count": len(self._queried)}
            self.last_info = info
            return Observation(messages=list(self._messages)), self.invalid_tool_penalty, False, info
        if not calls:
            info = {
                "error": "missing_submit_diagnosis",
                "query_count": len(self._queried),
            }
            self.last_info = info
            self._done = True
            return (
                Observation(messages=list(self._messages)),
                self.missing_submission_penalty,
                True,
                info,
            )

        # Calls in one generation are composed before any of their results are
        # visible to the policy. A final diagnosis must use a later generation.
        has_submission = any(call.name == "submit_diagnosis" for call in calls)
        error = None
        if has_submission and len(calls) != 1:
            error = "submit_must_be_separate_turn"
        elif has_submission and len(self._observed_views) < 2:
            error = "insufficient_observed_evidence"
        if error:
            response = json.dumps(
                {
                    "error": error,
                    "hint": "Query at least two distinct nonempty views, read the returned observations, "
                    "then call submit_diagnosis alone in a later turn.",
                }
            )
            self._messages.append(
                {
                    "role": "user",
                    "content": self.format_wrapper.format_observation(response),
                }
            )
            info = {
                "tool_calls": [{"tool": "submit_diagnosis", "error": error}],
                "query_count": len(self._queried),
            }
            self.last_info = info
            return (
                Observation(messages=list(self._messages)),
                self.invalid_tool_penalty,
                False,
                info,
            )

        # Validate the whole generation before executing any tool.
        for call in calls:
            args = call.arguments
            invalid = not isinstance(args, dict)
            if call.name in QUERY_TO_SECTION and not invalid:
                invalid = bool(set(args) - {"focus", "limit"})
                invalid |= "focus" in args and not isinstance(args["focus"], str)
                invalid |= "limit" in args and (type(args["limit"]) is not int or not 1 <= args["limit"] <= 50)
            elif call.name == "submit_diagnosis" and not invalid:
                required = {"root_causes", "evidence", "repair_actions", "confidence"}
                invalid = not required.issubset(args) or bool(set(args) - required - {"evidence_ids"})
                if not invalid:
                    invalid = any(
                        not isinstance(args[k], list)
                        or not args[k]
                        or any(not isinstance(v, str) or not v.strip() for v in args[k])
                        for k in ("root_causes", "evidence", "repair_actions")
                    )
                    invalid |= (
                        any(c not in ROOT_CAUSES for c in args["root_causes"])
                        if isinstance(args["root_causes"], list)
                        else True
                    )
                    confidence = args["confidence"]
                    invalid |= (
                        type(confidence) not in (int, float)
                        or not math.isfinite(confidence)
                        or not 0 <= confidence <= 1
                    )
                    refs = args.get("evidence_ids", [])
                    invalid |= not isinstance(refs, list) or any(
                        not isinstance(ref, str) or ref not in self.evidence_records for ref in refs
                    )
            if invalid:
                message = json.dumps({"error": "invalid_arguments", "tool": call.name})
                self._messages.append({"role": "user", "content": self.format_wrapper.format_observation(message)})
                info = {
                    "tool_calls": [{"tool": call.name, "error": "invalid_arguments"}],
                    "query_count": len(self._queried),
                }
                self.last_info = info
                return Observation(messages=list(self._messages)), self.invalid_tool_penalty, False, info

        total_reward = 0.0
        done = False
        observations: list[str] = []
        step_info: dict[str, Any] = {
            "tool_calls": [],
            "query_count": len(self._queried),
        }
        for call in calls:
            if done:
                break
            if call.name in QUERY_TO_SECTION:
                response, reward, info = self._query(call.name, call.arguments)
            elif call.name == "submit_diagnosis":
                response, reward, info = self._submit(call.arguments)
                done = True
            else:
                response = json.dumps(
                    {
                        "error": f"unknown tool: {call.name}",
                        "available_tools": list(QUERY_TO_SECTION) + ["submit_diagnosis"],
                    }
                )
                reward = self.invalid_tool_penalty
                info = {"tool": call.name, "error": "unknown_tool"}
            total_reward += float(reward)
            observations.append(self.format_wrapper.format_observation(response))
            step_info["tool_calls"].append(info)

        if observations:
            self._messages.append({"role": "user", "content": "\n".join(observations)})
        step_info["query_count"] = len(self._queried)
        self.last_info = step_info
        self._done = done
        return Observation(messages=list(self._messages)), total_reward, done, step_info
