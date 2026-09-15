# Training dependencies

The CPU protocol tests use `requirements-test.txt` only. The GPU trainer additionally requires a compatible PyTorch/CUDA, vLLM, Ray, Transformers, PEFT and verl environment.

Follow the source installation instructions preserved in `docs/upstream_README.md`, and `docs/getting-started/installation-guide.md`. The bundled upstream README specifies verl 0.7.0; validate its AgentFlow interfaces against this snapshot. Pin the exact versions used in each experiment; the preflight records installed versions but cannot establish end-to-end compatibility. It exits nonzero if required packages or CUDA are missing.

Do not install packages into an active shared training environment. Use a dedicated environment and select a device after checking occupancy. Start with configuration composition and a short train/dev run before scaling up. GPU or NPU training has not been validated in this public-release session.
