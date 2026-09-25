import argparse
import json
import random
import time

import opharm
import numpy as np
import torch

from opharm.analysis import POS, blast_partition, code, eval_split, load_rows, settings
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.interp import hooks
from opharm.interp.directions import unit, unmatched
from opharm.interp.probes import fit_probe
from opharm.models import load_model
from opharm.paths import RUNS
from opharm.run.decision import action_logodds


class Ctx:
    def __init__(self, model_key, tag, confirm):
        rows, self.acts = load_rows(model_key, tag, confirm)
        self.main = [r for r in rows if r["set"] == "main" and r["split"] == eval_split(confirm)]
        self.by_code = {(r["skeleton"], code(r)): r for r in self.main}
        self.tok, self.model = load_tokenizer(model_key), load_model(model_key)
        self.opener = self.tok.convert_tokens_to_ids("<tool_call>")
        st = settings(model_key, tag, confirm)
        self.blast, self.l_steer = {k: tuple(v) for k, v in st["blast"].items()}, st["L_steer"]
        dev_main = [r for r in rows if r["set"] == "main" and r["split"] == "dev"]
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


def sample_pairs(ctx, n, seed, first, second, pos):
    rng = random.Random(seed)
    cands = [(r, ctx.by_code[(r["skeleton"], code(r)[:pos] + second + code(r)[pos + 1:])])
             for r in ctx.main if code(r)[pos] == first]
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
            vals = ctx.capture_span(rs.ids, layers, positions)
            for layer in layers:
                m, p = ctx.run(rt.ids, [hooks.patch(ctx.model, layer, positions, vals[layer])], at=[rt.t_inst, rt.t_post])
                out.append({"target": tgt["id"], "source": src["id"], "layer": layer, "m_clean": m0, "m_source": ms, "m_patched": m,
                            "probe": {k: {"clean": p0[k], "source": ps[k], "patched": p[k]} for k in p0}})
    return out


def swap_panel(ctx, pairs, layers, names):
    dirs = {"r_blast": {l: torch.tensor(ctx.r_blast["t_inst"][l]) for l in layers},
            "r_blast_post": {l: torch.tensor(ctx.r_blast["t_post"][l]) for l in layers}}
    dirs["random"] = {l: random_like(dirs["r_blast"][l], 3000 + l) for l in layers}
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
    ap.add_argument("test", choices=["c1", "c2", "c3", "c4", "c6", "c7"])
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--layers", default="")
    ap.add_argument("--coefs", default="0.5,1,2")
    ap.add_argument("--groups", default="DP,DS,BP")
    ap.add_argument("--dirs", default="r_blast,random")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    t0 = time.time()
    ctx = Ctx(args.model, args.tag, args.confirm)
    n_layers = len(hooks.layers(ctx.model))
    layers = [int(v) for v in args.layers.split(",")] if args.layers else sorted({round(1 + i * (n_layers - 3) / 7) for i in range(8)})
    if args.test == "c1":
        out = patch_panel(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1), "env", layers)
    elif args.test == "c7":
        out = patch_panel(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1), "env", layers, complement=True)
    elif args.test == "c6":
        out = swap_panel(ctx, sample_pairs(ctx, args.n, 0, "P", "S", 1), layers, args.dirs.split(","))
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
        rows = [r for r in ctx.main if code(r)[:2] == "DP" and r["policy"] == "C" and r["label"] == "ASK"]
        out = ablate_panel(ctx, rows[: args.n], args.seeds)
    name = f"causal_{args.test}_{args.model}_{args.tag}{'_confirm' if args.confirm else ''}{'_' + args.suffix if args.suffix else ''}"
    (RUNS / args.model / args.tag / f"{name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    print(json.dumps({"test": args.test, "records": len(out), "layers": layers, "blast": ctx.blast, "L_steer": ctx.l_steer,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
