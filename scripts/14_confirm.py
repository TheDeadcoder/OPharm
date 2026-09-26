import argparse
import json
from collections import defaultdict

import opharm
import numpy as np
from sklearn.metrics import roc_auc_score

from opharm.analysis import POS, blast_partition, code, eval_split, load_rows, settings, user_text
from opharm.interp.directions import content_centroids, position
from opharm.interp.probes import fit_probe, ngram_scores
from opharm.paths import RESULTS, RUNS
from opharm.stats.bootstrap import cluster_draws, holm, p_beyond

MARGIN, ALPHA = 0.2, 0.05
STRATA = {}


def cls_of(skeleton):
    return STRATA.get(skeleton, skeleton.split(".")[0])


def paired(rows, value, pairs):
    key = {(r["skeleton"], code(r)): value[i] for i, r in enumerate(rows)}
    skels = sorted({r["skeleton"] for r in rows})
    return np.array([np.mean([key[(s, a)] - key[(s, b)] for a, b in pairs]) for s in skels]), [cls_of(s) for s in skels]


def estimate(per, strata, n):
    draws = np.asarray(per)[cluster_draws(strata, n)].mean(1)
    return float(np.mean(per)), [float(v) for v in np.percentile(draws, [2.5, 97.5])], draws, len(per)


def report(est, p, test):
    return {"estimate": est[0], "ci95": est[1], "n_skeletons": est[3], "test": test, "p": p}


def h1(rows, n):
    main = [r for r in rows if r["set"] == "main"]
    m = np.array([r["m"] for r in main])
    pol, strata = paired(main, m, [("DPAN", "DPAC"), ("DPNN", "DPNC")])
    blast, _ = paired(main, m, [("DSAN", "DPAN"), ("DSNN", "DPNN")])
    est = estimate(pol - blast, strata, n)
    out = report(est, p_beyond(est[2], 0.0, "greater"), "Delta_policy - Delta_blast > 0")
    out.update({"delta_policy": estimate(pol, strata, n)[:2], "delta_blast": estimate(blast, strata, n)[:2]})
    return out


def h2(rows, acts, st, cc, n):
    main = [r for r in rows if r["set"] == "main" and r["target"] == "D"]
    layer = st["L_harm"]
    x = acts[np.array([r["row"] for r in main]), 0, layer]
    pos = position(x, cc["benign"][0][layer], cc["harm"][0][layer])
    per, strata = paired(main, pos, [("DP" + rb + p, "DS" + rb + p) for rb in "AN" for p in "CN"])
    est = estimate(per, strata, n)
    p = max(p_beyond(est[2], MARGIN, "less"), p_beyond(est[2], -MARGIN, "greater"))
    return report(est, p, f"equivalence within +/-{MARGIN} (TOST)")


def h3(rows, acts, point, confirm, n):
    main = [r for r in rows if r["set"] == "main"]
    train, test, splits = blast_partition(main, confirm)
    env = np.array([r["env"] == "P" for r in main])
    skel = np.array([r["skeleton"] for r in main])
    x = acts[np.array([r["row"] for r in main]), POS[point[0]], point[1]]
    probe, _, c = fit_probe(x[train], env[train], skel[train], splits=splits)
    s_probe = probe.decision_function(x[test])
    texts = [user_text(r) for r in main]
    s_ngram = ngram_scores([t for t, k in zip(texts, train) if k], env[train], [t for t, k in zip(texts, test) if k])
    y, ts = env[test], skel[test]
    uniq = sorted(set(ts.tolist()))
    rows_of = [np.flatnonzero(ts == s) for s in uniq]

    def stat(ix):
        a = roc_auc_score(y[ix], s_probe[ix])
        return a - max(roc_auc_score(y[ix], v[ix]) for v in s_ngram.values()), a

    full = stat(np.arange(len(y)))
    draws = np.array([stat(np.concatenate([rows_of[k] for k in d]))[0] for d in cluster_draws([cls_of(s) for s in uniq], n)])
    est = (full[0], [float(v) for v in np.percentile(draws, [2.5, 97.5])], draws, len(uniq))
    out = report(est, p_beyond(draws, 0.0, "greater"), "AUROC(probe) - AUROC(best n-gram) > 0")
    out.update({"probe_auroc": full[1], "ngram_auroc": {k: float(roc_auc_score(y, v)) for k, v in s_ngram.items()},
                "C": c, "position": point[0], "layer": point[1], "n_test": int(test.sum())})
    return out


def h4(model, tag, confirm, st, n):
    path = RUNS / model / tag / f"causal_c3_{model}_{tag}{'_confirm_h4' if confirm else ''}.jsonl"
    if not path.exists():
        return None
    recs = [r for r in map(json.loads, open(path)) if r["coef"] == st["steer_coef"] and r["cell"] in ("DPAN", "DPNN")]
    shift = defaultdict(dict)
    for r in recs:
        shift[r["id"]][r["direction"]] = r["m"] - r["m_clean"]
    by_skel = defaultdict(list)
    for i, d in shift.items():
        by_skel[i.rsplit(".", 2)[0]].append(d["r_ref"] - d["r_blast"])
    skels = sorted(by_skel)
    est = estimate([np.mean(by_skel[s]) for s in skels], [cls_of(s) for s in skels], n)
    out = report(est, p_beyond(est[2], 0.0, "less"), "Delta_m(r_ref) - Delta_m(r_blast) < 0")
    mean = lambda pick: float(np.mean([v for d in shift.values() for k, v in d.items() if pick(k)]))
    out.update({"delta_m_r_ref": mean(lambda k: k == "r_ref"), "delta_m_r_blast": mean(lambda k: k == "r_blast"),
                "delta_m_random_ref": mean(lambda k: k.startswith("rand_ref")),
                "delta_m_random_blast": mean(lambda k: k.startswith("rand_blast")),
                "n_random_per_direction": len({k for d in shift.values() for k in d if k.startswith("rand_ref")})})
    return out


def h5(rows, acts, st, cc, n):
    ladder = [r for r in rows if r["set"] == "ladder"]
    layer = st["L_ref"]
    x = acts[np.array([r["row"] for r in ladder]), 1, layer]
    pos = position(x, cc["benign"][1][layer], cc["harm"][1][layer])
    key = {(r["skeleton"], r["rung"]): v for r, v in zip(ladder, pos)}
    skels = sorted({r["skeleton"] for r in ladder})
    est = estimate([key[(s, "harm")] - key[(s, "neutral")] for s in skels], [cls_of(s) for s in skels], n)
    return report(est, p_beyond(est[2], 0.0, "greater"), "Delta_ref > 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--strata", default="family", choices=["family", "class"])
    args = ap.parse_args()
    rows, acts = load_rows(args.model, args.tag, args.confirm)
    if args.strata == "class":
        STRATA.update({r["skeleton"]: r["cls"] for r in rows})
    st = settings(args.model, args.tag, args.confirm)
    split = eval_split(args.confirm)
    ev = [r for r in rows if r["split"] == split]
    cc = content_centroids(args.model)
    res = {"H1": h1(ev, args.n), "H2": h2(ev, acts, st, cc, args.n),
           "H3": h3(rows, acts, tuple(st["blast"]["registered"]), args.confirm, args.n),
           "H4": h4(args.model, args.tag, args.confirm, st, args.n), "H5": h5(ev, acts, st, cc, args.n)}
    done = [k for k, v in res.items() if v is not None]
    if args.confirm and len(done) < len(res):
        raise FileNotFoundError(f"missing {sorted(set(res) - set(done))}; run the H4 steering panel first")
    adj, rej = holm([res[k]["p"] for k in done], ALPHA)
    for k, a, r in zip(done, adj, rej):
        res[k].update({"p_holm": float(a), "reject_null": bool(r)})
    out = {"model": args.model, "tag": args.tag, "eval_split": split, "settings": st, "bootstrap_resamples": args.n,
           "family": done, "alpha": ALPHA, "hypotheses": res,
           "secondary": {"H3_dev_best": h3(rows, acts, tuple(st["blast"]["dev_best"]), args.confirm, args.n)}}
    RESULTS.mkdir(exist_ok=True)
    name = f"confirm_{args.model}_{args.tag}{'_confirm' if args.confirm else '_dev'}{'_class_strata' if args.strata == 'class' else ''}"
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (None if v is None else {q: v[q] for q in ("estimate", "ci95", "p", "p_holm", "reject_null")})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
