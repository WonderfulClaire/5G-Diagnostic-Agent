# Harness Generalization Evaluation

The original evaluation path uses the canonical TeleLogs tool names. This protocol adds a controlled way to test whether an agent has learned the diagnostic task or is overly dependent on one tool schema.

## What changes

Only the model-facing evidence-query tool names change.

The final submission contract stays fixed as submit_diagnosis, and the environment, case state, reward, step budget, prompt, and evaluator remain unchanged.

| Variant | Example query tools |
| --- | --- |
| canonical | query_radio_kpi, query_resource |
| compact | radio, resource |
| alternate | inspect_radio_metrics, inspect_resource_allocation |

Before environment execution, aliases are translated back to canonical names. The saved trace contains both raw_action and canonical action so schema sensitivity remains auditable.

## Run

~~~bash
python -m recipes.telelogs.evaluation.run_agent cases.jsonl \
  --base-url http://localhost:8000/v1 \
  --model MODEL_NAME \
  --max-steps 8 \
  --harness-variant canonical \
  --output runs/canonical.jsonl

python -m recipes.telelogs.evaluation.run_agent cases.jsonl \
  --base-url http://localhost:8000/v1 \
  --model MODEL_NAME \
  --max-steps 8 \
  --harness-variant compact \
  --output runs/compact.jsonl

python -m recipes.telelogs.evaluation.run_agent cases.jsonl \
  --base-url http://localhost:8000/v1 \
  --model MODEL_NAME \
  --max-steps 8 \
  --harness-variant alternate \
  --output runs/alternate.jsonl
~~~

Keep the same model checkpoint, case file, max steps, API backend, temperature, and evaluator.

## Recommended protocol

For post-training experiments:

~~~text
train / rollout harnesses:
    canonical + compact

held-out harness:
    alternate
~~~

Do not tune on alternate if it is reported as held-out.

Report at least:

- exact root-cause match
- mean sample set F1
- tool failure rate
- average tool-call count
- missing-submission rate
- step-budget exhaustion rate
- canonical-to-held-out performance gap

## Why this matters

Agent performance is a property of the model plus its interaction protocol. If the model performs well only with query_radio_kpi but degrades sharply when the same tool is exposed as inspect_radio_metrics, the result indicates harness sensitivity rather than pure diagnostic generalization.

This test does not prove robustness to arbitrary harnesses. It is a controlled schema-shift test with fixed semantics.

## Training claim boundary

A stronger post-training claim should therefore separate:

~~~text
task generalization
harness generalization
reward / verifier robustness
~~~

A model can improve on one while regressing on another. They should not be collapsed into one success number.
