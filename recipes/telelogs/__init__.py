"""Agent-R1 recipe for tool-augmented TeleLogs diagnosis."""

__all__ = ["AgentEnvLoop", "TeleLogsEnv"]


def __getattr__(name: str):
    if name == "AgentEnvLoop":
        # Importing the configured flow is also the point where the task
        # environment must be registered. Offline preprocessing/evaluation
        # modules can therefore remain usable without the heavyweight runtime.
        from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

        from .env import TeleLogsEnv  # noqa: F401

        return AgentEnvLoop
    if name == "TeleLogsEnv":
        from .env import TeleLogsEnv

        return TeleLogsEnv
    raise AttributeError(name)
