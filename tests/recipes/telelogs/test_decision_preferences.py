import torch
from recipes.telelogs.training.decision_preferences import preference_loss, decision_tokens


def test_preference_gradient_and_reference_are_correct():
    chosen = torch.tensor(-3., requires_grad=True)
    rejected = torch.tensor(-2., requires_grad=True)
    reference = torch.tensor(-3., requires_grad=True)
    loss = preference_loss(chosen, rejected, reference, torch.tensor(-2.))
    assert torch.allclose(loss, torch.tensor(2.).log())
    loss.backward()
    assert chosen.grad < 0 and rejected.grad > 0 and reference.grad is None


def test_token_boundaries_reconstruct_all_compared_strings():
    from types import SimpleNamespace
    import json
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(text.encode()))
    header = '{"root_causes": '
    alternatives = [["C1"], ["C1", "C8"]]
    prefix, actions = decision_tokens(Tokenizer(), header, alternatives)
    for a, expected in zip(actions, alternatives):
        assert bytes(prefix + a).decode() == header + json.dumps(expected) + ','
