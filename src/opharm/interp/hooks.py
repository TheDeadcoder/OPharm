from contextlib import contextmanager

import torch


def layers(model):
    return model.model.layers


def resid_modules(model):
    return list(layers(model)) + [model.model.norm]


def mixer(layer):
    return layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn


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
    post = [("post", mixer(l), fn) for l in layers(model)] + [("post", l.mlp, fn) for l in layers(model)]
    return pre + post
