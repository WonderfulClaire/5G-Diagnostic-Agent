# Third-party code and attribution

The general training framework in `agent_r1/`, upstream recipes, documentation and associated utilities originates from [Agent-R1](https://github.com/AgentR1/Agent-R1). This repository was prepared from the local source snapshot at commit `dff0328`; this short hash identifies the local starting snapshot, not a separately verified upstream release.

Original copyright notices and the upstream MIT [LICENSE](LICENSE) are retained. The bundled project-specific work is primarily in `recipes/telelogs/`, `examples/telelogs/`, the matching tests, and release tooling. Step-level RL, GRPO, LoRA and the upstream framework architecture are not claimed as original inventions of this project.

Additional dependencies, including verl, vLLM and pretrained models, retain their respective licenses. No gated telecom dataset is redistributed.
