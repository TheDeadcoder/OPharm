import argparse
import json
import random
import time
from collections import defaultdict

import opharm
import numpy as np
import torch

from opharm.analysis import POS, code, load_rows, settings
from opharm.bench.evaluate import evaluate
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.interp import hooks
from opharm.interp.directions import outlier_keep, unit, unmatched
from opharm.models import load_model
from opharm.paths import RUNS
from opharm.run.decision import _special, action_logodds, opener_ids
from opharm.run.generate import CLOSERS, end_ids, pad_id


def random_unit(u, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(u.shape, generator=g)
    r = r - (r @ u) * u
    return r / r.norm()


def displace(model, point, pos, delta):
    def fn(h):
        if h.shape[1] <= pos:
            return h
        h = h.clone()
        v = h[:, pos].float()
        h[:, pos] = (v + delta(v)).to(h.dtype)
        return h
    return [("pre", hooks.resid_modules(model)[point], fn)]


class Runner:
    def __init__(self, model_key, max_new):
        self.key, self.max_new = model_key, max_new
        self.tok, self.model = load_tokenizer(model_key), load_model(model_key)
        self.opener = opener_ids(self.tok)
        special = _special(self.tok)
        self.stops = sorted(end_ids(self.tok) | {self.tok.convert_tokens_to_ids(s) for s in CLOSERS if s in special})
        self.ends, self.pad = end_ids(self.tok), pad_id(self.tok)

    def render(self, r):
        return render(self.tok, r["system"], r["user"], TOOLS)

    def generate(self, ids, edits=()):
        x = torch.tensor([ids], device=self.model.device)
        with torch.inference_mode(), hooks.hooked(*edits):
            out = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=self.max_new, do_sample=False,
                                      eos_token_id=self.stops, pad_token_id=self.pad, output_logits=True, return_dict_in_generate=True)
        m = action_logodds(out.logits[0][0].float().cpu(), self.opener).item()
        row = out.sequences[0, len(ids):].tolist()
        cut = next((k for k, t in enumerate(row) if t in self.ends or t == self.pad), len(row))
        return m, self.tok.decode(row[:cut])

    def record(self, r, condition, edits=(), **extra):
        m, text = self.generate(self.render(r).ids, edits)
        return {"id": r["id"], "cell": code(r), "condition": condition, "m": m, "text": text,
                "label_v2": evaluate(text, r, self.key)["label"], **extra}

    def generate_many(self, id_lists, edits=()):
        n = max(len(ids) for ids in id_lists)
        x = torch.tensor([[self.pad] * (n - len(ids)) + ids for ids in id_lists], device=self.model.device)
        mask = torch.tensor([[0] * (n - len(ids)) + [1] * len(ids) for ids in id_lists], device=self.model.device)
        with torch.inference_mode(), hooks.hooked(*edits):
            out = self.model.generate(x, attention_mask=mask, max_new_tokens=self.max_new, do_sample=False,
                                      eos_token_id=self.stops, pad_token_id=self.pad, output_logits=True, return_dict_in_generate=True)
        res = []
        for i in range(len(id_lists)):
            row = out.sequences[i, n:].tolist()
            cut = next((k for k, t in enumerate(row) if t in self.ends or t == self.pad), len(row))
            res.append((action_logodds(out.logits[0][i].float().cpu(), self.opener).item(), self.tok.decode(row[:cut])))
        return res

    def records(self, rows, condition, edits, batch):
        order = sorted(rows, key=lambda r: len(self.render(r).ids))
        out = []
        for s in range(0, len(order), batch):
            part = order[s:s + batch]
            for r, (m, text) in zip(part, self.generate_many([self.render(r).ids for r in part], edits)):
                out.append({"id": r["id"], "cell": code(r), "condition": condition, "m": m, "text": text,
                            "label_v2": evaluate(text, r, self.key)["label"]})
        return out


def panel(dev_main, n_skeletons, seed):
    by_cls = defaultdict(set)
    for r in dev_main:
        by_cls[r["cls"]].add(r["skeleton"])
    rng = random.Random(seed)
    per_cls = max(1, n_skeletons // len(by_cls))
    return {s for skels in by_cls.values() for s in rng.sample(sorted(skels), min(per_cls, len(skels)))}


def c2(run, rows, st, n_random, limit=0, split="dev", masked=False, batch=1):
    dev_main = [r for r in rows if r["set"] == "main" and r["split"] == "dev"]
    pool = [r for r in rows if r["set"] == "main" and r["split"] == split]
    x = np.asarray(run.acts[[r["row"] for r in dev_main]])
    env = np.array([r["env"] == "P" for r in dev_main])
    skel = np.array([r["skeleton"] for r in dev_main])
    dirs = {f"r_blast_{k}": torch.tensor(unmatched(x[:, POS[p]], env, ~env, skel)[layer]) for k, (p, layer) in st["blast"].items()}
    dirs["r_ref"] = torch.tensor(np.load(RUNS / run.key / "refsets" / "directions.npz")["r_ref"][st["L_steer"]])
    base = dirs["r_blast_registered"]
    dirs.update({f"rand_{s}": random_unit(base / base.norm(), 2000 + s) * base.norm() for s in range(n_random)})
    asks = [r for r in pool if code(r)[:2] == "DP" and r["policy"] == "C" and r["label"] == "ASK"]
    asks = asks[:limit] if limit else asks
    keep = [torch.tensor(k) for k in outlier_keep(run.key)] if masked else None
    edit = lambda v: [hooks.ablate_masked(run.model, v, keep) if masked else hooks.ablate(run.model, v)]
    if batch > 1:
        return run.records(asks, "clean", [], batch) + [x for name, v in dirs.items() for x in run.records(asks, name, edit(v), batch)]
    out = []
    for r in asks:
        out.append(run.record(r, "clean"))
        for name, v in dirs.items():
            out.append(run.record(r, name, edit(v)))
    return out


def gain(run, rows, layer, gains, n_skeletons, seed, limit=0):
    dev_main = [r for r in rows if r["set"] == "main" and r["split"] == "dev"]
    keep = panel(dev_main, n_skeletons, seed)
    fit = [r for r in dev_main if r["skeleton"] not in keep]
    x = np.asarray(run.acts[[r["row"] for r in fit], POS["t_post"], layer], dtype=np.float32)
    env = np.array([r["env"] == "P" for r in fit])
    u = torch.tensor(unit(unmatched(x, env, ~env, np.array([r["skeleton"] for r in fit]))), dtype=torch.float32)
    proj = x @ u.numpy()
    center = {p: float(proj[np.array([r["env"] == "S" and r["policy"] == p for r in fit])].mean()) for p in "CN"}
    w = random_unit(u, 7000 + layer)
    test = sorted([r for r in dev_main if r["skeleton"] in keep], key=lambda r: r["id"])
    test = test[:limit] if limit else test
    amp = {r["id"]: float(np.asarray(run.acts[r["row"], POS["t_post"], layer], dtype=np.float32) @ u.numpy()) - center[r["policy"]]
           for r in test}
    rng, shuffled, mean_amp = random.Random(seed), {}, {}
    for p in "CN":
        ids = [r["id"] for r in test if r["policy"] == p]
        perm = ids[:]
        while len(ids) > 1 and any(a == b for a, b in zip(ids, perm)):
            rng.shuffle(perm)
        shuffled.update(zip(ids, perm))
        mean_amp[p] = float(np.mean([amp[i] for i in ids]))
    out = []
    for r in test:
        pos, a, c = run.render(r).t_post, amp[r["id"]], center[r["policy"]]
        out.append(run.record(r, "baseline", [], amplitude=a, displacement=0.0))
        out.append(run.record(r, "identity", [displace(run.model, layer, pos, lambda v: 0.0 * v)], amplitude=a, displacement=0.0))
        for g in gains:
            k = g - 1
            conds = {
                "gain": displace(run.model, layer, pos, lambda v, c=c, k=k: k * ((v @ u.to(v.device)) - c).unsqueeze(-1) * u.to(v.device)),
                "matched_random": displace(run.model, layer, pos, lambda v, c=c, k=k: k * ((v @ u.to(v.device)) - c).unsqueeze(-1) * w.to(v.device)),
                "shuffled_amplitude": displace(run.model, layer, pos, lambda v, b=amp[shuffled[r["id"]]], k=k: k * b * u.to(v.device).expand_as(v)),
                "fixed_addition": displace(run.model, layer, pos, lambda v, b=mean_amp[r["policy"]], k=k: k * b * u.to(v.device).expand_as(v)),
            }
            for name, edits in conds.items():
                out.append(run.record(r, f"{name}|{g:g}", [edits], amplitude=a, displacement=abs(k * a)))
    return out, {"layer": layer, "gains": gains, "center": center, "panel_skeletons": sorted(keep),
                 "fit_skeletons": len({r["skeleton"] for r in fit}), "mean_amplitude": mean_amp}


def steer(run, rows, st, coefs, n_random, n, seed, limit=0, split="dev", batch=1):
    dev_main = [r for r in rows if r["set"] == "main" and r["split"] == "dev"]
    pool = [r for r in rows if r["set"] == "main" and r["split"] == split]
    x = np.asarray(run.acts[[r["row"] for r in dev_main], POS["t_post"], st["L_steer"]], dtype=np.float32)
    env = np.array([r["env"] == "P" for r in dev_main])
    ref = torch.tensor(np.load(RUNS / run.key / "refsets" / "directions.npz")["r_ref"][st["L_steer"]])
    blast = torch.tensor(unit(unmatched(x, env, ~env, np.array([r["skeleton"] for r in dev_main])))) * ref.norm()
    dirs = {"r_ref": ref, "r_blast": blast, **{f"rand_{s}": random_unit(ref / ref.norm(), s) * ref.norm() for s in range(n_random)}}
    rng = random.Random(seed)
    picked = []
    for grp in ("DP", "DS", "BP"):
        cand = [r for r in pool if code(r) in {grp + "AN", grp + "NN"}]
        picked += rng.sample(cand, min(limit or n, n, len(cand)))
    edit = lambda v, cf: [hooks.steer(run.model, st["L_steer"], v, cf)]
    if batch > 1:
        return run.records(picked, "clean", [], batch) + [x for name, v in dirs.items() for cf in coefs
                                                         for x in run.records(picked, f"{name}|{cf}", edit(v, cf), batch)]
    out = []
    for r in picked:
        out.append(run.record(r, "clean"))
        for name, v in dirs.items():
            for cf in coefs:
                out.append(run.record(r, f"{name}|{cf}", edit(v, cf)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c2", "gain", "steer"])
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--gains", default="3,10")
    ap.add_argument("--skeletons", type=int, default=16)
    ap.add_argument("--random", type=int, default=3)
    ap.add_argument("--coefs", default="0.5,1")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    held = args.split == "heldout"
    rows, acts = load_rows(args.model, args.tag, held)
    run = Runner(args.model, args.max_new)
    run.acts = acts
    st = settings(args.model, args.tag, held)
    meta = {"model": args.model, "mode": args.mode, "args": vars(args)}
    if args.mode == "c2":
        out = c2(run, rows, st, args.random, args.limit, args.split, args.masked, args.batch)
    elif args.mode == "gain":
        layer = args.layer or round(0.82 * len(hooks.layers(run.model)))
        out, info = gain(run, rows, layer, [float(v) for v in args.gains.split(",")], args.skeletons, args.seed, args.limit)
        meta.update(info)
    else:
        out = steer(run, rows, st, [float(c) for c in args.coefs.split(",")], args.random, args.n, args.seed, args.limit, args.split, args.batch)
    dest = RUNS / args.model / args.tag / f"decoded_{args.mode}{args.suffix}.jsonl"
    dest.write_text("".join(json.dumps(o) + "\n" for o in out))
    meta.update(records=len(out), seconds=round(time.time() - t0, 1))
    (RUNS / args.model / args.tag / f"decoded_{args.mode}{args.suffix}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("mode", "records", "seconds")}))


if __name__ == "__main__":
    main()
