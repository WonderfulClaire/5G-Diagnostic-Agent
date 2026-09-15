# Agent-R1 × TeleLogs: multi-turn 5G root-cause diagnosis

This recipe converts the public TeleLogs task definition into a hidden-state,
tool-augmented Agent-R1 environment. The model initially sees only the
throughput-degradation symptom. It must query complementary engineering views
and finish with a structured diagnosis containing root causes, evidence,
repairs, and confidence.

## What is implemented

- Six diagnostic tools: radio KPI, cell relation/PCI, mobility, antenna,
  handover, and RB allocation.
- A seventh `submit_diagnosis` action that ends the episode.
- Step rewards for new evidence, repeated calls, invalid calls, grounded
  evidence, root-cause correctness, repair relevance, and tool efficiency.
- A gated-dataset adapter that keeps raw TeleLogs files out of Git.
- TeleLogsAgent FastAPI client plus TS1/TS2/TS3-compatible evaluation metrics.
- StepPO training and one-epoch smoke launchers for two GPUs.

## Data boundary

TeleLogs and TeleLogsAgent are gated benchmarks. Do not commit or redistribute
their samples. Accept each dataset's Hugging Face conditions and keep the token
in `HF_TOKEN`:

```bash
export HF_TOKEN=...  # keep this out of shell history when possible
python -m recipes.telelogs.data_preprocess.process_telelogs \
  --output-dir "$HOME/data/telelogs_agent_r1"
```

If the gated split files were downloaded through an authenticated browser,
keep them outside Git and preserve the official split names explicitly:

```bash
python -m recipes.telelogs.data_preprocess.process_telelogs \
  --local-train-file /secure/path/train.json \
  --local-test-file /secure/path/test.json \
  --output-dir "$HOME/data/telelogs_agent_r1"
```

The converter recognizes common `question`/`answer`/`label` layouts and fails
closed if it cannot normalize an example to the official C1-C8 taxonomy.

## Local checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
ruff check recipes/telelogs tests/recipes/telelogs
```

## GPU smoke run

```bash
DATA_DIR="$HOME/data/telelogs_agent_r1" \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
bash examples/telelogs/run_smoke.sh
```

For an auditable run that captures the Git state, dependency versions, GPU
inventory, configuration hashes, console log, and exit code:

```bash
bash scripts/telelogs/run_smoke_logged.sh
```

The smoke run is environment/training-pipeline evidence, not a final model
result. A resume-ready experiment should record at minimum:

1. Zero-shot/base-model exact match and macro F1.
2. StepPO exact match and macro F1 on the untouched TeleLogs test split.
3. TS1/TS2/TS3 task success, tool-call failure rate, average iterations, and
   average tool calls.
4. An ablation without process rewards and at least one qualitative trajectory.

When `trainer.validation_data_dir` is enabled, convert the raw step-level
trajectory dumps into task and tool-use metrics with:

```bash
python -m recipes.telelogs.evaluation.summarize_validation \
  --input /path/to/validation/dumps \
  --baseline-step 0 \
  --final-step 300 \
  --output-dir results/telelogs_validation
```

The resulting `metrics.json`, `report.md`, and
`trajectory_metrics.jsonl` keep the reported exact match, F1, submission,
evidence, repair, and tool-efficiency numbers traceable to raw trajectories.

## Official TeleLogsAgent server

After downloading the gated benchmark and starting its official server:

```bash
export TELELOGS_AGENT_CONFIG=TS1
python fastapi_server.py
python -m recipes.telelogs.evaluation.telelogs_agent_client \
  --scenario-id <scenario-id> --endpoint scenario
```

The client follows the official `X-Scenario-Id` header and endpoint names. It
does not copy benchmark samples into this repository.

Fill [`reports/telelogs/experiment_report.md`](../../reports/telelogs/experiment_report.md)
only from generated artifacts; its `TBD` placeholders are deliberate claim
gates.
