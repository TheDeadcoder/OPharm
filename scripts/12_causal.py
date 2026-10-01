import argparse
import json
import random
import time

import opharm
import numpy as np
import torch

from opharm.analysis import POS, blast_partition, code, eval_split, load_rows, settings
from opharm.bench.labels import final_label
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.interp import hooks
from opharm.interp.directions import unit, unmatched
from opharm.interp.probes import fit_probe
from opharm.models import load_model
from opharm.paths import RESULTS, RUNS
from opharm.run.decision import action_logodds, opener_ids


class Ctx:
    def __init__(self, model_key, tag, confirm):
        rows, self.acts = load_rows(model_key, tag, confirm)
        self.main = [r for r in rows if r["set"] == "main" and r["split"] == eval_split(confirm)]
        self.by_code = {(r["skeleton"], code(r)): r for r in self.main}
        self.tok, self.model = load_tokenizer(model_key), load_model(model_key)
        self.key, self.tag = model_key, tag
        self.inputs, self.limit = hooks.has_layer_inputs(self.model), hooks.last_patchable_layer(self.model)
        self.opener = opener_ids(self.tok)
        st = settings(model_key, tag, confirm)
        self.blast, self.l_steer = {k: tuple(v) for k, v in st["blast"].items()}, st["L_steer"]
        dev_main = [r for r in rows if r["set"] == "main" and r["split"] == "dev"]
        self.dev_main = dev_main
        x = np.asarray(self.acts[[r["row"] for r in dev_main]])
        env = np.array([r["env"] == "P" for r in dev_main])
        skel = np.array([r["skeleton"] for r in dev_main])
        self.r_blast = {p: unmatched(x[:, i], env, ~env, skel) for p, i in POS.items()}
        train, _, splits = blast_partition(dev_main, False)
        self.probes = {k: fit_probe(x[train, POS[p], layer], env[train], skel[train], splits=splits)[0]
                       for k, (p, layer) in self.blast.items()}
        self.r_ref = np.load(RUNS / model_key / "refsets" / "directions.npz")["r_ref"]

    def render(self, r):
        return render(self.tok, r["system"], r["user"], TOOLS)

    def run(self, ids, edits=(), at=None):
        store, extra = {}, []
        if at is not None:
            extra = hooks.capture(self.model, sorted({layer for _, layer in self.blast.values()}), at, store)
        with torch.inference_mode(), hooks.hooked(*edits, extra):
            logits = self.model(input_ids=torch.tensor([ids], device=self.model.device), use_cache=False, logits_to_keep=1).logits[0, -1]
        m = action_logodds(logits.float().cpu(), self.opener).item()
        if at is None:
            return m, None
        return m, {k: float(self.probes[k].decision_function(store[layer][0, POS[p]][None].numpy())[0])
                   for k, (p, layer) in self.blast.items()}

    def capture_span(self, ids, layers, positions):
        store = {}
        with torch.inference_mode(), hooks.hooked(hooks.capture(self.model, layers, positions, store)):
            self.model(input_ids=torch.tensor([ids], device=self.model.device), use_cache=False, logits_to_keep=1)
        return {layer: store[layer][0] for layer in layers}

    def capture_inputs(self, ids):
        if not self.inputs:
            return None
        store = {}
        with torch.inference_mode(), hooks.hooked(hooks.capture_layer_inputs(self.model, store)):
            self.model(input_ids=torch.tensor([ids], device=self.model.device), use_cache=False, logits_to_keep=1)
        return store

    def patch_edits(self, layer, positions, values, inputs):
        edits = [hooks.patch(self.model, layer, positions, values)]
        return edits + ([hooks.patch_layer_inputs(self.model, layer, positions, inputs)] if inputs is not None else [])


def sample_pairs(ctx, n, seed, first, second, pos, policy=""):
    rng = random.Random(seed)
    cands = [(r, ctx.by_code[(r["skeleton"], code(r)[:pos] + second + code(r)[pos + 1:])])
             for r in ctx.main if code(r)[pos] == first and (not policy or r["policy"] == policy)]
    rng.shuffle(cands)
    return cands[:n]


def patch_panel(ctx, pairs, span, layers, complement=False):
    out = []
    for a, b in pairs:
        for tgt, src in ((a, b), (b, a)):
            rt, rs = ctx.render(tgt), ctx.render(src)
            positions = [i for s, e in rt.spans[span] for i in range(s, e)]
            if complement:
                positions = [i for i in range(len(rt.ids)) if i not in set(positions)]
            m0, p0 = ctx.run(rt.ids, at=[rt.t_inst, rt.t_post])
            ms, ps = ctx.run(rs.ids, at=[rs.t_inst, rs.t_post])
            vals, inputs = ctx.capture_span(rs.ids, layers, positions), ctx.capture_inputs(rs.ids)
            for layer in layers:
                m, p = ctx.run(rt.ids, ctx.patch_edits(layer, positions, vals[layer], inputs), at=[rt.t_inst, rt.t_post])
                out.append({"target": tgt["id"], "source": src["id"], "layer": layer, "m_clean": m0, "m_source": ms, "m_patched": m,
                            "probe": {k: {"clean": p0[k], "source": ps[k], "patched": p[k]} for k in p0}})
    return out


def decomposition(ctx, layers):
    rows = ctx.dev_main
    x = np.asarray(ctx.acts[[r["row"] for r in rows], POS["t_post"]], dtype=np.float32)
    skel = np.array([r["skeleton"] for r in rows])
    masks = [np.array([f(r) for r in rows]) for f in (lambda r: r["target"] == "D", lambda r: r["rollback"] == "N", lambda r: r["policy"] == "C")]
    factors = [unmatched(x, m, ~m, skel) for m in masks]
    out = {"r_blast_shared": {}, "r_blast_unique": {}, "pc1": {}}
    for l in layers:
        span = torch.linalg.qr(torch.tensor(np.stack([f[l] / np.linalg.norm(f[l]) for f in factors], 1), dtype=torch.float32))[0]
        u = torch.tensor(unit(ctx.r_blast["t_post"][l]), dtype=torch.float32)
        out["r_blast_shared"][l] = span @ (span.T @ u)
        out["r_blast_unique"][l] = u - out["r_blast_shared"][l]
        h = x[:, l].astype(np.float64)
        out["pc1"][l] = torch.tensor(np.linalg.svd(h - h.mean(0), full_matrices=False)[2][0], dtype=torch.float32)
    return out


def swap_panel(ctx, pairs, layers, names):
    dirs = {"r_blast": {l: torch.tensor(ctx.r_blast["t_inst"][l]) for l in layers},
            "r_blast_post": {l: torch.tensor(ctx.r_blast["t_post"][l]) for l in layers}}
    dirs["random"] = {l: random_like(dirs["r_blast"][l], 3000 + l) for l in layers}
    if {"r_blast_shared", "r_blast_unique", "pc1"} & set(names):
        dirs.update(decomposition(ctx, layers))
    dirs = {k: dirs[k] for k in names}
    out = []
    for a, b in pairs:
        for tgt, src in ((a, b), (b, a)):
            rt, rs = ctx.render(tgt), ctx.render(src)
            m0, p0 = ctx.run(rt.ids, at=[rt.t_inst, rt.t_post])
            ms, ps = ctx.run(rs.ids, at=[rs.t_inst, rs.t_post])
            span = {i for s, e in rt.spans["env"] for i in range(s, e)}
            keep = [i for i in range(len(rt.ids)) if i not in span]
            vals = ctx.capture_span(rs.ids, layers, keep)
            for name, d in dirs.items():
                for layer in layers:
                    u = d[layer] / d[layer].norm()
                    m, p = ctx.run(rt.ids, [hooks.swap_along(ctx.model, layer, u, vals[layer] @ u, keep)], at=[rt.t_inst, rt.t_post])
                    out.append({"target": tgt["id"], "source": src["id"], "layer": layer, "direction": name, "m_clean": m0,
                                "m_source": ms, "m_patched": m,
                                "probe": {k: {"clean": p0[k], "source": ps[k], "patched": p[k]} for k in p0}})
    return out


def gain_panel(ctx, rows, layers, gains, seeds):
    staging = np.array([r["row"] for r in ctx.dev_main if r["env"] == "S" and r["policy"] == "N"])
    points = [("t_post", layer) for layer in layers]
    if ctx.blast["dev_best"][0] == "t_inst":
        points.append(("t_inst", ctx.blast["dev_best"][1]))
    specs = []
    for p_name, layer in points:
        u = torch.tensor(unit(ctx.r_blast[p_name][layer]), dtype=torch.float32)
        dirs = {"r_blast": u, **{f"rand_{s}": random_like(u, 4000 + 100 * layer + s) for s in range(seeds)}}
        proj = np.asarray(ctx.acts[staging, POS[p_name], layer], dtype=np.float32)
        for name, v in dirs.items():
            v = v / v.norm()
            specs.append((p_name, layer, name, v, float((proj @ v.numpy()).mean())))
    out = []
    for r in rows:
        rr = ctx.render(r)
        m0, _ = ctx.run(rr.ids)
        for p_name, layer, name, v, center in specs:
            pos = [rr.t_post if p_name == "t_post" else rr.t_inst]
            for g in gains:
                m, _ = ctx.run(rr.ids, [hooks.gain(ctx.model, layer, v, center, g, pos)])
                out.append({"id": r["id"], "cell": code(r), "position": p_name, "layer": layer, "direction": name,
                            "gain": g, "m_clean": m0, "m": m})
    return out


def patch_check(ctx, pairs, layers):
    out = {"layers": layers, "pairs": len(pairs), "per_layer": {}}
    for layer in layers:
        ident, trans, resid_only = [], [], []
        for a, b in pairs:
            ra, rb = ctx.render(a), ctx.render(b)
            every = list(range(len(ra.ids)))
            ma, _ = ctx.run(ra.ids)
            mb, _ = ctx.run(rb.ids)
            own, src = ctx.capture_span(ra.ids, [layer], every), ctx.capture_span(rb.ids, [layer], every)
            own_in, src_in = ctx.capture_inputs(ra.ids), ctx.capture_inputs(rb.ids)
            ident.append(abs(ctx.run(ra.ids, ctx.patch_edits(layer, every, own[layer], own_in))[0] - ma))
            trans.append(abs(ctx.run(ra.ids, ctx.patch_edits(layer, every, src[layer], src_in))[0] - mb))
            resid_only.append(abs(ctx.run(ra.ids, ctx.patch_edits(layer, every, src[layer], None))[0] - mb))
        out["per_layer"][layer] = {"identity_max": max(ident), "transplant_max": max(trans), "residual_only_transplant_max": max(resid_only)}
    out["valid"] = all(v["identity_max"] < 1e-3 and v["transplant_max"] < 1e-2 for v in out["per_layer"].values())
    return out


def random_like(v, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    u = v / v.norm()
    r = r - (r @ u) * u
    return r / r.norm() * v.norm()


def steer_panel(ctx, rows, coefs, seeds):
    ref = torch.tensor(ctx.r_ref[ctx.l_steer])
    u_ref = ref / ref.norm()
    blast = torch.tensor(unit(ctx.r_blast["t_post"][ctx.l_steer])) * ref.norm()
    perp = blast - (blast @ u_ref) * u_ref
    dirs = {"r_ref": ref, "r_blast": blast, "r_blast_perp": perp / perp.norm() * ref.norm(),
            "r_blast_inst": torch.tensor(unit(ctx.r_blast["t_inst"][ctx.l_steer])) * ref.norm()}
    dirs.update({f"rand_ref_{s}": random_like(ref, s) for s in range(seeds)})
    dirs.update({f"rand_blast_{s}": random_like(blast, 1000 + s) for s in range(seeds)})
    out = []
    for r in rows:
        ids = ctx.render(r).ids
        m0, _ = ctx.run(ids)
        for name, v in dirs.items():
            for c in coefs:
                m, _ = ctx.run(ids, [hooks.steer(ctx.model, ctx.l_steer, v, c)])
                out.append({"id": r["id"], "cell": code(r), "direction": name, "coef": c, "m_clean": m0, "m": m})
    return out


def ablate_panel(ctx, rows, seeds):
    dirs = {f"r_blast_{k}": torch.tensor(ctx.r_blast[p][layer]) for k, (p, layer) in ctx.blast.items()}
    dirs["r_ref"] = torch.tensor(ctx.r_ref[ctx.l_steer])
    dirs.update({f"rand_{s}": random_like(dirs["r_blast_registered"], 2000 + s) for s in range(seeds)})
    out = []
    for r in rows:
        ids = ctx.render(r).ids
        m0, _ = ctx.run(ids)
        for name, v in dirs.items():
            m, _ = ctx.run(ids, [hooks.ablate(ctx.model, v)])
            out.append({"id": r["id"], "cell": code(r), "direction": name, "m_clean": m0, "m": m})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["c1", "c2", "c3", "c4", "c6", "c7", "c8", "check"])
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--layers", default="")
    ap.add_argument("--coefs", default="0.5,1,2")
    ap.add_argument("--groups", default="DP,DS,BP")
    ap.add_argument("--dirs", default="r_blast,random")
    ap.add_argument("--gains", default="3,10,30")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--policy", default="", choices=["", "C", "N"])
    args = ap.parse_args()
    t0 = time.time()
    ctx = Ctx(args.model, args.tag, args.confirm)
    n_layers = len(hooks.layers(ctx.model))
    layers = [int(v) for v in args.layers.split(",")] if args.layers else sorted({round(1 + i * (n_layers - 3) / 7) for i in range(8)})
    if args.test in ("c1", "c4", "c6", "c7", "check") and ctx.limit is not None and max(layers) > ctx.limit:
        raise ValueError(f"{args.model} shares key/value states after layer {ctx.limit}; patch layers {layers} go past it")
    if args.test == "check":
        res = patch_check(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1, args.policy), layers)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"patchcheck_{args.model}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    if args.test == "c1":
        out = patch_panel(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1, args.policy), "env", layers)
    elif args.test == "c8":
        rng = random.Random(3)
        rows = []
        for g in ("DP", "DS", "BP", "BS"):
            cand = [r for r in ctx.main if code(r) in {g + "AN", g + "NN"}]
            rows += rng.sample(cand, min(args.n, len(cand)))
        late = [int(v) for v in args.layers.split(",")] if args.layers else sorted({round(f * n_layers) for f in (0.7, 0.82, 0.94)})
        out = gain_panel(ctx, rows, late, [float(g) for g in args.gains.split(",")], args.seeds)
    elif args.test == "c7":
        out = patch_panel(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1, args.policy), "env", layers, complement=True)
    elif args.test == "c6":
        out = swap_panel(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1, args.policy), layers, args.dirs.split(","))
    elif args.test == "c4":
        out = patch_panel(ctx, sample_pairs(ctx, args.n, 1, "C", "N", 3), "policy", layers)
    elif args.test == "c3":
        rng = random.Random(2)
        rows = []
        for g in args.groups.split(","):
            cand = [r for r in ctx.main if code(r) in {g + "AN", g + "NN"}]
            rows += rng.sample(cand, min(args.n, len(cand)))
        out = steer_panel(ctx, rows, [float(c) for c in args.coefs.split(",")], args.seeds)
    else:
        rows = [r for r in ctx.main if code(r)[:2] == "DP" and r["policy"] == "C" and final_label(r, args.model, args.tag)["label"] == "ASK"]
        out = ablate_panel(ctx, rows[: args.n], args.seeds)
    name = f"causal_{args.test}_{args.model}_{args.tag}{'_confirm' if args.confirm else ''}{'_' + args.suffix if args.suffix else ''}"
    (RUNS / args.model / args.tag / f"{name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    print(json.dumps({"test": args.test, "records": len(out), "layers": layers, "blast": ctx.blast, "L_steer": ctx.l_steer,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
