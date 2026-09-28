import opharm
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from opharm.interp import hooks


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=32)
    return LlamaForCausalLM(cfg).eval()


def run(model, edits):
    x = torch.randint(0, 64, (1, 7), generator=torch.Generator().manual_seed(1))
    store = {}
    points = list(range(len(hooks.resid_modules(model))))
    with torch.inference_mode(), hooks.hooked(*edits, hooks.capture(model, points, None, store)):
        logits = model(input_ids=x).logits
    return logits, store


def test_all_kept_matches_ablate():
    model = tiny()
    d = torch.randn(16)
    keep = [torch.ones(16, dtype=torch.bool) for _ in hooks.resid_modules(model)]
    a, _ = run(model, [hooks.ablate(model, d)])
    b, _ = run(model, [hooks.ablate_masked(model, d, keep)])
    assert torch.allclose(a, b, atol=1e-5)


def test_masked_dimension_untouched():
    model = tiny()
    d = torch.randn(16)
    keep = [torch.ones(16, dtype=torch.bool) for _ in hooks.resid_modules(model)]
    for k in keep:
        k[3] = False
    _, base = run(model, [])
    _, edited = run(model, [[e for e in hooks.ablate_masked(model, d, keep) if e[0] == "pre"][:1]])
    assert torch.allclose(base[0][..., 3], edited[0][..., 3])
    u = d * keep[0].float()
    u = u / u.norm()
    assert torch.allclose(edited[0] @ u, torch.zeros(1, 7), atol=1e-5)
