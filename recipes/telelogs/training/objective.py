"""GRPO objective for action tokens only; observations are context, never targets."""

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


def learning_route(rewards, correctness, epsilon=1e-6):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    correctness = torch.tensor(correctness, dtype=torch.float32)
    rs = float(rewards.std(unbiased=False))
    qs = float(correctness.std(unbiased=False))
    if bool((correctness == 1).all()):
        return "efficiency_rl" if rs > epsilon else "mastered_replay"
    if qs > epsilon and rs > epsilon:
        return "rl_ready"
    if qs > epsilon:
        return "repair_reward_resolution"
    return "audit_reward_only_variance" if rs > epsilon else "teacher_or_sft_repair"
