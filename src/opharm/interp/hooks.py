from contextlib import contextmanager

import torch


def decoder(model):
    return getattr(model.model, "language_model", model.model)


def layers(model):
    return decoder(model).layers


def resid_modules(model):
    return list(layers(model)) + [decoder(model).norm]


def mixer(layer):
    return layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn


def writers(layer):
    if hasattr(layer, "pre_feedforward_layernorm") and hasattr(layer, "post_feedforward_layernorm"):
        mods = [layer.post_attention_layernorm, layer.post_feedforward_layernorm]
        return mods + ([layer.post_per_layer_input_norm] if hasattr(layer, "post_per_layer_input_norm") else [])
    return [mixer(layer), layer.mlp]


def _idx(positions):
    return slice(None) if positions is None else positions


def _on_input(fn):
    def hook(module, args):
        return (fn(args[0]),) + tuple(args[1:])
    return hook


def _on_output(fn):
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (fn(output[0]),) + tuple(output[1:])
        return fn(output)
    return hook


@contextmanager
def hooked(*edits):
    handles = []
    try:
        for group in edits:
            for kind, module, fn in group:
                hook = _on_input(fn) if kind == "pre" else _on_output(fn)
                register = module.register_forward_pre_hook if kind == "pre" else module.register_forward_hook
                handles.append(register(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def capture(model, points, positions, store):
    mods = resid_modules(model)

    def make(p):
        def fn(h):
            store[p] = h[:, _idx(positions)].detach().float().cpu()
            return h
        return fn

    return [("pre", mods[p], make(p)) for p in points]


def patch(model, point, positions, values):
    def fn(h):
        h = h.clone()
        h[:, _idx(positions)] = values.to(device=h.device, dtype=h.dtype)
        return h

    return [("pre", resid_modules(model)[point], fn)]


def steer(model, point, vector, alpha, positions=None):
    def fn(h):
        h = h.clone()
        idx = _idx(positions)
        h[:, idx] = (h[:, idx].float() + alpha * vector.to(h.device, torch.float32)).to(h.dtype)
        return h

    return [("pre", resid_modules(model)[point], fn)]


def ablate(model, direction, positions=None):
    unit = direction.float() / direction.float().norm()

    def fn(h):
        h = h.clone()
        idx = _idx(positions)
        u = unit.to(h.device)
        sub = h[:, idx].float()
        h[:, idx] = (sub - (sub @ u).unsqueeze(-1) * u).to(h.dtype)
        return h

    pre = [("pre", m, fn) for m in resid_modules(model)]
    post = [("post", w, fn) for l in layers(model) for w in writers(l)]
    return pre + post


def swap_along(model, point, direction, values, positions=None):
    unit = direction.float() / direction.float().norm()

    def fn(h):
        h = h.clone()
        idx = _idx(positions)
        u = unit.to(h.device)
        sub = h[:, idx].float()
        h[:, idx] = (sub + (values.to(h.device)[None] - sub @ u).unsqueeze(-1) * u).to(h.dtype)
        return h

    return [("pre", resid_modules(model)[point], fn)]


def gain(model, point, direction, center, g, positions=None):
    unit = direction.float() / direction.float().norm()

    def fn(h):
        h = h.clone()
        idx = _idx(positions)
        u = unit.to(h.device)
        sub = h[:, idx].float()
        h[:, idx] = (sub + (g - 1) * ((sub @ u) - center).unsqueeze(-1) * u).to(h.dtype)
        return h

    return [("pre", resid_modules(model)[point], fn)]
