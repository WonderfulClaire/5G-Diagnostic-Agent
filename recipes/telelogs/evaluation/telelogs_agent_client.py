"""Small stdlib client for the official TeleLogsAgent FastAPI tools."""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OFFICIAL_ENDPOINTS = (
    "scenario",
    "signaling-plane-event-log",
    "throughput-logs",
    "cell-info",
    "gnodeb-location",
    "user-location",
    "user-speed",
    "serving-cell-pci",
    "serving-cell-rsrp",
    "serving-cell-sinr",
    "rbs-allocated-to-user",
    "neighboring-cells-pci",
    "neighboring-cell-rsrp",
    "beam-scenario-info",
    "tools",
)


class TeleLogsAgentHTTPClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7861", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, endpoint: str, scenario_id: str, **params: Any) -> Any:
        endpoint = endpoint.strip("/")
        if endpoint not in OFFICIAL_ENDPOINTS:
            raise ValueError(f"Unsupported endpoint {endpoint!r}; expected one of {OFFICIAL_ENDPOINTS}")
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.base_url}/{endpoint}{query}",
            headers={"Accept": "application/json", "X-Scenario-Id": scenario_id},
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - caller controls the local server URL
            payload = response.read().decode("utf-8")
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7861")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--endpoint", choices=OFFICIAL_ENDPOINTS, default="scenario")
    args = parser.parse_args()
    result = TeleLogsAgentHTTPClient(args.base_url).call(args.endpoint, args.scenario_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
