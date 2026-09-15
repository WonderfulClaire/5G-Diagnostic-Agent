"""GRPO objective for action tokens only; observations are context, never targets."""

import math
import torch


def group_advantages(rewards, epsilon=1e-6):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Need >=2 finite group rewards")
    std = rewards.std(unbiased=False)
    return (rewards - rewards.mean()) / std.clamp_min(epsilon), float(std)


def action_loss(new_logp, old_logp, reference_logp, advantage, clip=0.2, beta=0.01):
    if new_logp.ndim != 1 or new_logp.numel() == 0:
        raise ValueError("Expected nonempty action token log probabilities")
    ratio = (new_logp - old_logp.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = reference_logp.detach() - new_logp
    kl = torch.exp(delta) - delta - 1
    return (-surrogate + beta * kl).sum(), {
        "kl_sum": float(kl.detach().sum()),
        "clip_count": int(((ratio - 1).abs() > clip).sum()),
        "tokens": len(new_logp),
    }



def efficiency_route(rewards, costs, epsilon):
    if max(rewards) - min(rewards) <= epsilon:
        return "mastered_replay"
    if costs is None or max(costs) - min(costs) <= epsilon:
        return "audit_reward_only_variance"
    conflict = any(
        abs(costs[i] - costs[j]) > epsilon and abs(rewards[i] - rewards[j]) > epsilon
        and (costs[i] - costs[j]) * (rewards[i] - rewards[j]) > 0
        for i in range(len(rewards)) for j in range(i)
    )
    aligned = any(
        abs(costs[i] - costs[j]) > epsilon and abs(rewards[i] - rewards[j]) > epsilon
        and (costs[i] - costs[j]) * (rewards[i] - rewards[j]) < 0
        for i in range(len(rewards)) for j in range(i)
    )
    return "audit_reward_efficiency_conflict" if conflict else (
        "efficiency_rl" if aligned else "audit_reward_only_variance")

def learning_route(rewards, correctness, epsilon=1e-6, *, efficiency_costs=None):
    if efficiency_costs is not None and (
        len(efficiency_costs) != len(rewards) or
        any(not math.isfinite(float(c)) or c < 0 for c in efficiency_costs)
    ):
        raise ValueError("Need aligned finite nonnegative efficiency costs")

    if len(rewards) < 2 or len(rewards) != len(correctness):
        raise ValueError("Need aligned rollout groups")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if any(not math.isfinite(float(x)) for x in [*rewards, *correctness]):
        raise ValueError("Nonfinite rollout scores")
    if any(not 0 <= x <= 1 for x in correctness):
        raise ValueError("Correctness must be between 0 and 1")
    # Reward variance alone can actively teach worse decisions. Reject a group
    # if any resolvable quality ordering is reversed by the optimization reward.
    inversions = any(
        abs(correctness[i] - correctness[j]) > epsilon
        and abs(rewards[i] - rewards[j]) > epsilon
        and (correctness[i] - correctness[j]) * (rewards[i] - rewards[j]) < 0
        for i in range(len(rewards)) for j in range(i)
    )
    rewards = torch.tensor(rewards, dtype=torch.float32)
    correctness = torch.tensor(correctness, dtype=torch.float32)
    rs = float(rewards.std(unbiased=False))
    qs = float(correctness.std(unbiased=False))
    if bool((correctness == 1).all()):
        return efficiency_route(rewards.tolist(), efficiency_costs, epsilon)
    if qs > epsilon and rs > epsilon:
        return "audit_reward_quality_conflict" if inversions else "rl_ready"
    if qs > epsilon:
        return "repair_reward_resolution"
    return "audit_reward_only_variance" if rs > epsilon else "teacher_or_sft_repair"
