import argparse
import json
import time

import opharm
import numpy as np
from sklearn.metrics import roc_auc_score

from opharm.analysis import POS, TRAIN_CLASSES, TRAIN_FORMS, blast_partition, code, eval_split, load_rows, locked, user_text
from opharm.chat import snapshot_dir
from opharm.interp.directions import content_centroids, position, unit, unmatched
from opharm.interp.probes import fit_probe, ngram_auroc
from opharm.paths import RESULTS
from opharm.stats.bootstrap import cluster_ci

ENV_PAIRS_D = [("DP" + rb + p, "DS" + rb + p) for rb in "AN" for p in "CN"]


def paired(values, rows, pairs):
    key = {(r["skeleton"], code(r)): i for i, r in enumerate(rows)}
    skels = sorted({r["skeleton"] for r in rows})
    per = np.stack([np.mean([values[key[(s, a)]] - values[key[(s, b)]] for a, b in pairs], axis=0) for s in skels])
    cls = [s.split(".")[0] for s in skels]
    return per, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rows, acts = load_rows(args.model, args.tag, args.confirm)
    split = eval_split(args.confirm)
    main_rows = [r for r in rows if r["set"] == "main"]
    x = np.asarray(acts[[r["row"] for r in main_rows]])
    n_points = x.shape[2]
    ev = np.array([r["split"] == split for r in main_rows])
    ev_rows = [r for r in main_rows if r["split"] == split]
    env = np.array([r["env"] == "P" for r in main_rows])
    skel = np.array([r["skeleton"] for r in main_rows])
    labels = {"env": env, "rollback": np.array([r["rollback"] == "N" for r in main_rows]),
              "target": np.array([r["target"] == "D" for r in main_rows]),
              "danger": np.array([r["target"] == "D" and r["env"] == "P" for r in main_rows])}
    out = {"model": args.model, "tag": args.tag, "n_main": len(main_rows), "eval_split": split, "confirm": args.confirm}

    cc = content_centroids(args.model)
    if args.confirm:
        lk = locked(args.model)
        l_harm, l_ref = lk["L_harm"], lk["L_ref"]
    else:
        ref_summary = json.loads((RESULTS / f"refsets_{args.model}.json").read_text())
        l_harm = int(np.argmax(ref_summary["auroc_eval"]["r_harm"]))
        l_ref = int(np.argmax(ref_summary["auroc_eval"]["r_ref"]))
    pos_harm = position(x[ev, 0], cc["benign"][0], cc["harm"][0])
    pos_ref = position(x[ev, 1], cc["benign"][1], cc["harm"][1])
    seen = {"L_harm": l_harm, "L_ref": l_ref}
    for name, pos, layer in (("harm_t_inst", pos_harm, l_harm), ("ref_t_post", pos_ref, l_ref)):
        per, cls = paired(pos, ev_rows, ENV_PAIRS_D)
        est = cluster_ci(per[:, layer], cls)
        seen[name] = {"delta_env_on_destructive": {"estimate": est[0], "ci95": est[1:]},
                      "delta_by_layer": per.mean(0).round(4).tolist(),
                      "absolute_by_cell": {c: float(np.mean([pos[i, layer] for i, r in enumerate(ev_rows) if code(r)[:2] == c]))
                                           for c in ("DP", "DS", "BP", "BS")}}
    out["seen"] = seen

    train, test, splits = blast_partition(main_rows, args.confirm)
    scan = {}
    for p_name, p in POS.items():
        scan[p_name] = []
        for layer in range(1, n_points, args.layer_step):
            model, cv_auc, c = fit_probe(x[train, p, layer], env[train], skel[train], splits=splits)
            test_auc = roc_auc_score(env[test], model.decision_function(x[test, p, layer]))
            scan[p_name].append({"layer": layer, "cv_auroc": cv_auc, "test_auroc": float(test_auc), "C": c})
    if args.confirm:
        points = {k: tuple(v) for k, v in lk["blast"].items()}
    else:
        reg = max(scan["t_post"], key=lambda d: d["cv_auroc"])
        p_best, best = max(((p_name, d) for p_name, ds in scan.items() for d in ds), key=lambda t: t[1]["test_auroc"])
        points = {"registered": ("t_post", reg["layer"]), "dev_best": (p_best, best["layer"])}
    texts = [user_text(r) for r in main_rows]
    tr_texts, te_texts = [t for t, m in zip(texts, train) if m], [t for t, m in zip(texts, test) if m]
    represented = {"rules": {"registered": "argmax leave-one-form-out AUROC within training at t_post",
                             "dev_best": "argmax development doubly held-out AUROC over positions and layers"},
                   "scan": scan, "n_train": int(train.sum()), "n_test": int(test.sum()),
                   "ngram_env": ngram_auroc(tr_texts, env[train], te_texts, env[test]), "points": {}}
    masked_rows = [r for r in rows if r["set"] == "masked" and r["split"] == split
                   and r["cls"] not in TRAIN_CLASSES and r["form"] not in TRAIN_FORMS]
    for name, (p_name, layer) in points.items():
        pi, at = POS[p_name], next(d for d in scan[p_name] if d["layer"] == layer)
        factor = {}
        for f, y in labels.items():
            model, cv_auc, c = fit_probe(x[train, pi, layer], y[train], skel[train], splits=splits)
            factor[f] = {"cv_auroc": cv_auc, "test_auroc": float(roc_auc_score(y[test], model.decision_function(x[test, pi, layer]))),
                         "ngram": ngram_auroc(tr_texts, y[train], te_texts, y[test])}
        point = {"position": p_name, "layer": layer, "C": at["C"], "cv_auroc": at["cv_auroc"], "test_auroc": at["test_auroc"],
                 "factors": factor}
        if masked_rows:
            env_probe, _, _ = fit_probe(x[train, pi, layer], env[train], skel[train], splits=splits)
            s_mask = env_probe.decision_function(np.asarray(acts[[r["row"] for r in masked_rows], pi, layer]))
            s_main = env_probe.decision_function(x[test, pi, layer])
            p_mean, s_mean = s_main[env[test]].mean(), s_main[~env[test]].mean()
            point["masked_position_between_staging_0_and_production_1"] = float((s_mask.mean() - s_mean) / (p_mean - s_mean))
        represented["points"][name] = point
    out["represented"] = represented

    judge = [r for r in rows if r["set"] == "judge" and r["question"] == "q2" and r["split"] == "dev"]
    if judge:
        xj = np.asarray(acts[[r["row"] for r in judge], 1])
        pj = np.array([r["env"] == "P" for r in judge])
        r_judg = unmatched(xj, pj, ~pj, [r["skeleton"] for r in judge])
        out["judgment_direction_readout_auroc_by_layer"] = [
            float(roc_auc_score(env[ev], x[ev, 1, layer] @ unit(r_judg[layer]))) for layer in range(1, n_points)]

    r_blast = {p: unmatched(x[ev, i], env[ev], ~env[ev], skel[ev]) for p, i in POS.items()}
    r_harm, r_ref = cc["harm"][0] - cc["benign"][0], cc["harm"][1] - cc["benign"][1]
    cos = lambda a, b: (unit(a) * unit(b)).sum(-1).round(4).tolist()
    geometry = {"cos_blast_harm_t_inst": cos(r_blast["t_inst"], r_harm), "cos_blast_ref_t_post": cos(r_blast["t_post"], r_ref)}
    flat = [r for r in rows if r["set"] == "flat" and r["split"] == split]
    if flat:
        idx = {r["id"]: r["row"] for r in ev_rows}
        pairs = [(idx[r["id"].replace(".flat.", ".main.")], r["row"]) for r in flat if r["id"].replace(".flat.", ".main.") in idx]
        r_schema = np.asarray(acts[[a for a, _ in pairs], 1]).mean(0) - np.asarray(acts[[b for _, b in pairs], 1]).mean(0)
        geometry.update({"cos_schema_ref_t_post": cos(r_schema, r_ref), "cos_schema_blast_t_post": cos(r_schema, r_blast["t_post"])})
    risky = [i for i, r in enumerate(main_rows) if ev[i] and code(r)[:2] == "DP" and r["policy"] == "C" and r["label"] in ("ASK", "EXEC_MATCH")]
    asked = np.array([main_rows[i]["label"] == "ASK" for i in risky])
    if asked.sum() >= 10 and (~asked).sum() >= 10:
        r_ask = x[risky][asked, 1].mean(0) - x[risky][~asked, 1].mean(0)
        geometry.update({"n_asked": int(asked.sum()), "cos_ask_ref_t_post": cos(r_ask, r_ref)})
    out["geometry"] = geometry
    cfg = json.loads((snapshot_dir(args.model) / "config.json").read_text())
    types = cfg.get("text_config", cfg).get("layer_types") or ["full_attention"] * (n_points - 1)
    out["point_block_type"] = ["attention" if t == "full_attention" else "gdn" for t in types] + ["final"]
    out["seconds"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"representation_{args.model}_{args.tag}{'_confirm' if args.confirm else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"seen": {k: (v["delta_env_on_destructive"] if isinstance(v, dict) else v) for k, v in seen.items()},
                      "ngram_env": represented["ngram_env"],
                      "points": {k: {q: v for q, v in d.items() if q != "factors"} for k, d in represented["points"].items()},
                      "factors": {k: {f: (v["test_auroc"], max(v["ngram"].values())) for f, v in d["factors"].items()}
                                  for k, d in represented["points"].items()},
                      "judg_readout_max": max(out.get("judgment_direction_readout_auroc_by_layer", [0])),
                      "geometry": {k: {q: (v[layer] if isinstance(v, list) else v) for q, v in geometry.items()}
                                   for k, (_, layer) in points.items()},
                      "seconds": out["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
