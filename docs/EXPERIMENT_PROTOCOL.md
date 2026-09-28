# 5G diagnosis experiment protocol

Keep environment, query budgets, tools and evaluation fixed across baseline and trained policies. Use a train-derived development split for checkpoint selection, and exclude every test case and its near-duplicates from training and augmentation.

Compare: untrained instruction model; LoRA-SFT if demonstrations exist; LoRA-GRPO; GRPO without novelty bonus; GRPO without repetition/call-cost penalties. Match total rollout and token budgets, use at least three seeds, and save model revision, dataset hashes, runtime versions and raw trajectories.

Report exact match, mean per-case set F1, micro F1, query count, invalid/repeated/empty calls, completion rate, referenced-evidence availability and manually checked causal support. Separate tool errors, missing evidence, unsupported explanation and root-cause errors. Keyword-based reward is only a training proxy.

Release criteria: no protocol regression; an independent evaluation shows diagnosis quality is preserved or improved while query cost decreases; confidence intervals and failed cases are retained. A passed smoke test, reward rise or completed optimizer step does not meet this criterion by itself.


## Offline reward-alignment audit

Every GRPO run should preserve `metrics.jsonl` and pass a post-run audit before its result is used in a report:

```bash
python -m scripts.telelogs.audit_reward_alignment runs/grpo/metrics.jsonl --strict
```

The audit recomputes `learning_route` from the saved reward, correctness and efficiency-cost vectors instead of trusting the stored route. It flags:

- stored route != recomputed route;
- optimizer update on a group routed to reward-quality / reward-efficiency audit;
- skipped optimizer update on a group that was actually `rl_ready` or `efficiency_rl`;
- malformed or non-finite records.

This is a training-integrity check, not a substitute for independent reward validation. A reward can still be consistently wrong; the audit only verifies that the trainer respected its own routing contract.


## Paired checkpoint uncertainty

Validation reports now support a deterministic paired bootstrap over the same held-out cases. The trainer persists a checkpoint-stable `case_key` derived from immutable validation input + ground truth, and the summarizer can reconstruct the same key for older dumps.

```bash
python -m recipes.telelogs.evaluation.summarize_validation \
  --input runs/validation \
  --baseline-step 0 \
  --final-step 300 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 0
```

The report gives the baseline, final checkpoint, observed delta, and a 95% paired-bootstrap interval for exact match, sample-set F1, micro F1, reward, query cost, tool-failure rate, submission rate, and evidence-related metrics.

The interval is descriptive uncertainty on this fixed validation set. It does not replace multi-seed training runs, and it should not be treated as evidence for a different dataset or deployment population.
