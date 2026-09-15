# Release validation — 2026-09-15

Python 3.12, local CPU. `python -m pytest tests/recipes/telelogs -q`: 22 passed in 0.15 s.

Validated: atomic query/submit separation; at least two nonempty observations; observation-only model context; content-hashed new/repeated evidence; nonzero query cost; invalid argument/submission rejection; episode reset; harness step budget; trace serialization; root-cause set metrics.

The OpenAI-compatible request adapter is implemented; tests use controlled backend responses and do not establish model-service compatibility or diagnosis quality. LoRA-GRPO GPU execution and independent model quality evaluation are pending.

Historical training reports are kept locally and are not republished as current-release evidence. The earlier local report recorded unchanged diagnosis quality despite reward growth; it is not used to claim an improvement here.
