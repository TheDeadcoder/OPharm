from types import SimpleNamespace

import torch
from torch import nn

from opharm.interp import hooks


class Layer(nn.Module):
    def forward(self, h, x, **kwargs):
        return h + 10 * x.sum(-1, keepdim=True)


class Toy(nn.Module):
    def __init__(self, n=4, shared=0, types=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Layer() for _ in range(n))
        self.model.norm = nn.Identity()
        self.config = SimpleNamespace(hidden_size_per_layer_input=2, num_hidden_layers=n, num_kv_shared_layers=shared,
                                      layer_types=types or ["sliding_attention"] * n)

    def forward(self, h, x):
        for i, layer in enumerate(self.model.layers):
            h = layer(h, x[:, :, i])
        return h


def test_capture_and_patch_layer_inputs():
    torch.manual_seed(0)
    model, h = Toy(), torch.zeros(1, 3, 1)
    a, b = torch.randn(1, 3, 4, 2), torch.randn(1, 3, 4, 2)
    store = {}
    with hooks.hooked(hooks.capture_layer_inputs(model, store)):
        model(h, b)
    assert sorted(store) == [0, 1, 2, 3] and torch.equal(store[2], b[:, :, 2])
    with hooks.hooked(hooks.patch_layer_inputs(model, 2, [1], store)):
        out = model(h, a)
    x = a.clone()
    x[:, 1, 2:] = b[:, 1, 2:]
    assert torch.allclose(out, model(h, x))
    assert not torch.allclose(out, model(h, a))


def test_patch_limit_from_kv_sharing():
    kinds = (["sliding_attention"] * 5 + ["full_attention"]) * 7
    assert hooks.last_patchable_layer(Toy(42, 18, kinds)) == 22
    assert hooks.last_patchable_layer(Toy(4, 0)) is None
    assert hooks.has_layer_inputs(Toy())
