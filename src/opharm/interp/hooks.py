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
                if kind == "raw":
                    handles.append(module.register_forward_pre_hook(fn))
                    continue
                hook = _on_input(fn) if kind == "pre" else _on_output(fn)
                register = module.register_forward_pre_hook if kind == "pre" else module.register_forward_hook
                handles.append(register(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def has_layer_inputs(model):
    return bool(getattr(text_config(model), "hidden_size_per_layer_input", 0))


def last_patchable_layer(model):
    cfg = text_config(model)
    shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
    if not shared:
        return None
    types = cfg.layer_types[: cfg.num_hidden_layers - shared]
    return min(max(i for i, t in enumerate(types) if t == kind) for kind in set(types))


def capture_layer_inputs(model, store):
    def make(i):
        def hook(module, args):
            store[i] = args[1].detach().clone()
        return hook
    return [("raw", layer, make(i)) for i, layer in enumerate(layers(model))]


def patch_layer_inputs(model, start, positions, values):
    def make(i):
        def hook(module, args):
            x = args[1].clone()
            x[:, _idx(positions)] = values[i][:, _idx(positions)].to(device=x.device, dtype=x.dtype)
            return (args[0], x) + tuple(args[2:])
        return hook
    return [("raw", layers(model)[i], make(i)) for i in range(start, len(layers(model)))]


def capture(model, points, positions, store):
    mods = resid_modules(model)

    def make(p):
        def fn(h):
            store[p] = h[:, _idx(positions)].detach().float().cpu()
            return h
        return fn

    return [("pre", mods[p], make(p)) for p in points]


def ablate_masked(model, direction, keep, positions=None):
    def make(mask):
        unit = direction.float() * mask.float()
        unit = unit / unit.norm()

        def fn(h):
            h = h.clone()
            idx = _idx(positions)
            u = unit.to(h.device)
            sub = h[:, idx].float()
            h[:, idx] = (sub - (sub @ u).unsqueeze(-1) * u).to(h.dtype)
            return h
        return fn

    pre = [("pre", m, make(keep[i])) for i, m in enumerate(resid_modules(model))]
    post = [("post", w, make(keep[l + 1])) for l, layer in enumerate(layers(model)) for w in writers(layer)]
    return pre + post


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
