import argparse
import json
from collections import defaultdict

import opharm
import numpy as np

from opharm.paths import RESULTS, RUNS
from opharm.stats.bootstrap import cluster_ci, cluster_ratio_ci


def load(model, tag, test, confirm, suffix=""):
    path = RUNS / model / tag / f"causal_{test}_{model}_{tag}{'_confirm' if confirm else ''}{'_' + suffix if suffix else ''}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def by_skeleton(recs, value):
    per = defaultdict(list)
    for r in recs:
        per[r.get("target", r.get("id")).rsplit(".", 2)[0]].append(value(r))
    keys = sorted(per)
    return np.array([np.mean(per[k]) for k in keys]), [k.split(".")[0] for k in keys]


def mean_ci(recs, value):
    return [round(x, 3) for x in cluster_ci(*by_skeleton(recs, value))]


def ratio_ci(recs, num, den):
    (a, strata), (b, _) = by_skeleton(recs, num), by_skeleton(recs, den)
    return [round(x, 3) for x in cluster_ratio_ci(a, b, strata)] if abs(b.mean()) > 1e-6 else None


def patch_summary(recs, pos):
    groups = defaultdict(list)
    for r in recs:
        t, s = r["target"].rsplit(".", 1)[1][pos], r["source"].rsplit(".", 1)[1][pos]
        groups[(r["layer"], f"{t}<-{s}" + (f"|{r['direction']}" if "direction" in r else ""))].append(r)
    out = {}
    for (layer, d), rs in sorted(groups.items()):
        gap, moved = (lambda x: x["m_source"] - x["m_clean"]), (lambda x: x["m_patched"] - x["m_clean"])
        entry = {"n": len(rs), "m_gap": mean_ci(rs, gap), "m_moved": mean_ci(rs, moved), "m_fraction": ratio_ci(rs, moved, gap)}
        for k in rs[0]["probe"]:
            pgap = lambda x, k=k: x["probe"][k]["source"] - x["probe"][k]["clean"]
            pmoved = lambda x, k=k: x["probe"][k]["patched"] - x["probe"][k]["clean"]
            entry[f"probe_{k}"] = {"gap": mean_ci(rs, pgap), "fraction": ratio_ci(rs, pmoved, pgap)}
        out[f"{layer}|{d}"] = entry
    return out


def steer_summary(recs):
    groups = defaultdict(list)
    for r in recs:
        kind = r["direction"] if not r["direction"].startswith("rand") else "_".join(r["direction"].split("_")[:2])
        groups[(kind, r["coef"], r["cell"][:2])].append(r)
    table = {f"{k}|{c}|{cell}": mean_ci(rs, lambda x: x["m"] - x["m_clean"]) for (k, c, cell), rs in sorted(groups.items())}
    select = {}
    for kind in sorted({k for k, _, _ in groups if not k.startswith("rand")}):
        for c in sorted({r["coef"] for r in recs}):
            dp, ds, bp = (table[f"{kind}|{c}|{cell}"][0] if f"{kind}|{c}|{cell}" in table else np.nan for cell in ("DP", "DS", "BP"))
            select[f"{kind}|{c}"] = {"selectivity_env": round(float(ds - dp), 3), "selectivity_target": round(float(bp - dp), 3)}
    return {"mean_m_shift": table, "selectivity": select}


def ablate_summary(recs):
    groups = defaultdict(list)
    for r in recs:
        groups["random" if r["direction"].startswith("rand") else r["direction"]].append(r)
    return {k: {"n": len(v), "m_shift": mean_ci(v, lambda x: x["m"] - x["m_clean"]),
                "flip_to_act": mean_ci(v, lambda x: float(x["m_clean"] < 0 < x["m"]))} for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    out = {}
    env = lambda r: patch_summary(r, 1)
    for test, suffix, fn in (("c1", "", env), ("c4", "", lambda r: patch_summary(r, 3)), ("c3", "", steer_summary),
                             ("c2", "", ablate_summary), ("c6", "", env), ("c6", "post", env), ("c7", "", env)):
        recs = load(args.model, args.tag, test, args.confirm, suffix)
        if recs:
            out[test + ("_" + suffix if suffix else "")] = fn(recs)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"causal_{args.model}_{args.tag}{'_confirm' if args.confirm else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
