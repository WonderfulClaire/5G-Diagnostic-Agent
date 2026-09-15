import torch
from recipes.telelogs.training.objective import group_advantages, action_loss


def test_equal_rewards_have_no_policy_gradient():
    advantage, std = group_advantages([0.2] * 4)
    assert std == 0 and torch.equal(advantage, torch.zeros(4))


def test_advantage_sign_and_frozen_targets():
    advantage, std = group_advantages([0.0, 1.0])
    x = torch.tensor([-1.0, -2.0], requires_grad=True)
    old = x.detach().clone().requires_grad_()
    ref = x.detach().clone().requires_grad_()
    loss, _ = action_loss(x, old, ref, advantage[1])
    loss.backward()
    assert (x.grad < 0).all() and old.grad is None and ref.grad is None


def test_proxy_only_variance_is_not_a_quality_learning_signal():
    from recipes.telelogs.training.objective import learning_route

    assert learning_route([0.7, 0.8], [2 / 3, 2 / 3]) == "audit_reward_only_variance"
    assert learning_route([0.1, 0.9], [0, 1]) == "rl_ready"


def test_only_response_positions_contribute_to_training_loss():
    from types import SimpleNamespace
    from recipes.telelogs.training.online_grpo import log_probs

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 4, 8))

        def forward(self, ids):
            return SimpleNamespace(logits=self.logits[:, : ids.shape[1]])

    model = TinyPolicy()
    prefix = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])
    scores = log_probs(model, prefix, response)
    assert scores.shape == (2,)
    (-scores.mean()).backward()
    assert torch.equal(model.logits.grad[:, :2], torch.zeros(1, 2, 8))
    assert model.logits.grad[:, 2:].abs().sum() > 0
