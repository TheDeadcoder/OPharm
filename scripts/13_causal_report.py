import argparse
import json
from collections import defaultdict

import opharm
import numpy as np

from opharm.paths import RESULTS, RUNS


def load(model, tag, test, confirm):
    path = RUNS / model / tag / f"causal_{test}_{model}_{tag}{'_confirm' if confirm else ''}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def patch_summary(recs, pos):
    groups = defaultdict(list)
    for r in recs:
        t, s = r["target"].rsplit(".", 1)[1][pos], r["source"].rsplit(".", 1)[1][pos]
        groups[(r["layer"], f"{t}<-{s}")].append(r)
    frac = lambda moved, gap: moved / gap if abs(gap) > 1e-6 else None
    out = {}
    for (layer, d), rs in sorted(groups.items()):
        mg, mm = (float(np.mean([x[k] - x["m_clean"] for x in rs])) for k in ("m_source", "m_patched"))
        entry = {"n": len(rs), "m_gap": mg, "m_moved": mm, "m_fraction": frac(mm, mg)}
        for k in rs[0]["probe"]:
            pg, pm = (float(np.mean([x["probe"][k][q] - x["probe"][k]["clean"] for x in rs])) for q in ("source", "patched"))
            entry[f"probe_{k}"] = {"gap": pg, "moved": pm, "fraction": frac(pm, pg)}
        out[f"{layer}|{d}"] = entry
    return out


def steer_summary(recs):
    groups = defaultdict(list)
    for r in recs:
        kind = r["direction"] if not r["direction"].startswith("rand") else "_".join(r["direction"].split("_")[:2])
        groups[(kind, r["coef"], r["cell"][:2])].append(r["m"] - r["m_clean"])
    table = {f"{k}|{c}|{cell}": float(np.mean(v)) for (k, c, cell), v in sorted(groups.items())}
    select = {}
    for kind in sorted({k for k, _, _ in groups if not k.startswith("rand")}):
        for c in sorted({r["coef"] for r in recs}):
            dp, ds, bp = (np.mean(groups[(kind, c, cell)]) if groups[(kind, c, cell)] else np.nan for cell in ("DP", "DS", "BP"))
            select[f"{kind}|{c}"] = {"caution_DP": float(-dp), "caution_DS": float(-ds), "caution_BP": float(-bp),
                                     "selectivity_env": float(-dp + ds), "selectivity_target": float(-dp + bp)}
    return {"mean_m_shift": table, "selectivity": select}


def ablate_summary(recs):
    groups = defaultdict(list)
    for r in recs:
        kind = "random" if r["direction"].startswith("rand") else r["direction"]
        groups[kind].append(r)
    return {k: {"n": len(v), "m_shift_mean": float(np.mean([r["m"] - r["m_clean"] for r in v])),
                "flip_to_act": float(np.mean([(r["m_clean"] < 0) and (r["m"] > 0) for r in v]))} for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    out = {}
    for test, fn in (("c1", lambda r: patch_summary(r, 1)), ("c4", lambda r: patch_summary(r, 3)), ("c3", steer_summary),
                     ("c2", ablate_summary)):
        recs = load(args.model, args.tag, test, args.confirm)
        if recs:
            out[test] = fn(recs)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"causal_{args.model}_{args.tag}{'_confirm' if args.confirm else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
