#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs}"
RUN_NAME="${RUN_NAME:-telelogs-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

bash "$PROJECT_DIR/scripts/telelogs/capture_run_metadata.sh" "$RUN_DIR"

set +e
bash "$PROJECT_DIR/examples/telelogs/run_smoke.sh" "$@" 2>&1 | tee "$RUN_DIR/console.log"
STATUS=${PIPESTATUS[0]}
set -e

printf '%s\n' "$STATUS" >"$RUN_DIR/exit_code.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" >"$RUN_DIR/finished_at_utc.txt"
exit "$STATUS"
