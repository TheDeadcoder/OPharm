import os

import pytest
import torch

from opharm.chat import load_tokenizer, render
from opharm.interp import hooks
from opharm.models import last_logits, load_model

KEY = os.environ.get("OPHARM_TEST_MODEL", "qwen35_08b")


@pytest.fixture(scope="module")
def setup():
    tok = load_tokenizer(KEY)
    model = load_model(KEY)

    def encode(env):
        r = render(tok, [(None, "You are the operations agent.")],
                   [(None, "Run `df -h` on db-07.\nEnvironment:"), ("env", env), (None, "\nPlease handle this ticket.")])
        return torch.tensor([r.ids], device=model.device)

    a, b = encode(" production"), encode(" staging")
    assert a.shape == b.shape
    return model, a, b


def test_deterministic(setup):
    model, a, _ = setup
    assert torch.equal(last_logits(model, a), last_logits(model, a))


def test_noop_hooks(setup):
    model, a, _ = setup
    base = last_logits(model, a)
    with hooks.hooked([("pre", m, lambda h: h) for m in hooks.resid_modules(model)]):
        assert torch.equal(last_logits(model, a), base)


def test_self_patch_and_zero_steer(setup):
    model, a, _ = setup
    base = last_logits(model, a)
    store = {}
    with hooks.hooked(hooks.capture(model, [5], None, store)):
        last_logits(model, a)
    with hooks.hooked(hooks.patch(model, 5, None, store[5])):
        assert torch.equal(last_logits(model, a), base)
    with hooks.hooked(hooks.steer(model, 5, torch.randn(store[5].shape[-1]), 0.0)):
        assert torch.equal(last_logits(model, a), base)


def test_full_patch_reproduces_source(setup):
    model, a, b = setup
    target = last_logits(model, b)
    n = len(hooks.layers(model))
    for point in (0, 3, n // 2, n - 1):
        store = {}
        with hooks.hooked(hooks.capture(model, [point], None, store)):
            last_logits(model, b)
        with hooks.hooked(hooks.patch(model, point, None, store[point])):
            assert torch.equal(last_logits(model, a), target)


def test_ablation_removes_direction_everywhere(setup):
    model, a, _ = setup
    d = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    u = d / d.norm()
    n = len(hooks.layers(model))
    store = {}
    with hooks.hooked(hooks.ablate(model, d), hooks.capture(model, [1, n // 2, n], None, store)):
        logits = last_logits(model, a)
    assert torch.isfinite(logits).all()
    for h in store.values():
        assert (h @ u).abs().max() < 1e-2 * h.norm(dim=-1).max()
