#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <run-directory>" >&2
    exit 2
fi

RUN_DIR="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$RUN_DIR"

date -u +"%Y-%m-%dT%H:%M:%SZ" >"$RUN_DIR/started_at_utc.txt"
git -C "$PROJECT_DIR" rev-parse HEAD >"$RUN_DIR/git_commit.txt"
git -C "$PROJECT_DIR" status --short >"$RUN_DIR/git_status.txt"
git -C "$PROJECT_DIR" diff --binary >"$RUN_DIR/uncommitted.patch"
python3 --version >"$RUN_DIR/python_version.txt" 2>&1
python3 -m pip freeze >"$RUN_DIR/pip_freeze.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"$RUN_DIR/nvidia_smi.txt"
    nvidia-smi -q >"$RUN_DIR/nvidia_smi_q.txt"
fi

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum \
        "$PROJECT_DIR/recipes/telelogs/base.yaml" \
        "$PROJECT_DIR/examples/telelogs/run_steppo.sh" \
        "$PROJECT_DIR/examples/telelogs/run_smoke.sh" \
        >"$RUN_DIR/config_sha256.txt"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 \
        "$PROJECT_DIR/recipes/telelogs/base.yaml" \
        "$PROJECT_DIR/examples/telelogs/run_steppo.sh" \
        "$PROJECT_DIR/examples/telelogs/run_smoke.sh" \
        >"$RUN_DIR/config_sha256.txt"
fi

echo "captured run metadata in $RUN_DIR"
