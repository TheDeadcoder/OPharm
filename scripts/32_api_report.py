import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict

import opharm
import numpy as np
import yaml

from opharm.analysis import rule_coverage_map
from opharm.bench.evaluate import VERSION
from opharm.bench.labels import label
from opharm.paths import BENCH, CONFIGS, RESULTS, ROOT, RUNS
from opharm.stats.bootstrap import cluster_ratio_ci
from opharm.stats.lock import analysis_rows

sys.path.insert(0, str(ROOT / "scripts"))
ti = importlib.import_module("28_template_intervals")
pr = importlib.import_module("29_pairing")

CELLS, OTHER_CELLS, MIN_TEMPLATES = ("DP", "DS", "BP", "BS"), ("DS", "BP", "BS"), 5
NOTES = {"direct": "direct system prompt, low thinking", "narration": "narration system prompt, low thinking",
         "thinking": "direct system prompt, high thinking"}
EXEC, ASK, interval = ti.EXEC, ti.ASK, ti.interval


def cell(r):
    return r["target"] + r["env"]


def key(r):
    return r["skeleton"], cell(r) + r["rollback"] + r["policy"]


def load(model, tags, meta, split):
    rows = []
    for tag in tags:
        path = RUNS / model / tag / "results.jsonl"
        for r in map(json.loads, open(path)) if path.exists() else ():
            m = meta.get(r["id"])
            if m is None or (split != "all" and m["split"] != split):
                continue
            row = {**m, **r}
            if m["set"] != "judge":
                row["label_v2"] = label(r["text"], m, model, None)["label"]
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{model} {tags}: the same instance appears in more than one run")
    return analysis_rows(rows)


def examples(rows, k=3):
    out = defaultdict(list)
    for r in rows:
        if r["label_v2"] not in ("EXEC_MATCH", "ASK") and len(out[r["label_v2"]]) < k:
            out[r["label_v2"]].append({"id": r["id"], "reply_tail": r["text"][-300:]})
    return dict(out)


def summary(rows, cover):
    pick = lambda c, p: [r for r in rows if cell(r) == c and r["policy"] == p]
    free = [r for r in rows if r["policy"] == "C" and cell(r) in OTHER_CELLS]
    covered = [r for r in pick("DP", "C") if cover.get(r["template"]) == "Y"]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["policy"] == "C":
            by[r["skeleton"]]["need" if cell(r) == "DP" else "free"].append(float(ASK(r)))
    gaps = [{"skeleton": s, "v": float(np.mean(d["need"]) - np.mean(d["free"]))} for s, d in by.items() if d["need"] and d["free"]]
    sel = ti.selectivity(rows)
    return {"n": len(rows), "examples_other_labels": examples(rows),
            "cells": {c + p: {"exec": interval(pick(c, p), EXEC), "ask": interval(pick(c, p), ASK),
                              "labels": dict(Counter(r["label_v2"] for r in pick(c, p)))} for p in "CN" for c in CELLS},
            "covered_DPC": {"exec": interval(covered, EXEC), "ask": interval(covered, ASK)},
            "over_ask": interval(free, ASK), "executes_where_allowed": interval(free, EXEC),
            "ask_selectivity": interval(gaps, lambda r: r["v"]),
            "ask_env_selectivity": sel["ask_env"], "paired_correct_abstention": sel["paired_correct_abstention"]}


def versus(direct, rows):
    base = {key(r): r for r in direct}
    both = [{"skeleton": r["skeleton"], "policy": r["policy"], "cell": cell(r), "x": base[key(r)], "y": r} for r in rows if key(r) in base]
    diff = lambda f: (lambda p: float(f(p["y"])) - float(f(p["x"])))
    out = {"n_matched": len(both), "covers_all_direct_prompts": len(both) == len(direct)}
    for p in "CN":
        for c in CELLS:
            sub = [b for b in both if b["cell"] == c and b["policy"] == p]
            out[c + p] = {"exec_difference": interval(sub, diff(EXEC)), "ask_difference": interval(sub, diff(ASK))}
    out["over_ask_difference"] = interval([b for b in both if b["policy"] == "C" and b["cell"] in OTHER_CELLS], diff(ASK))
    return out


def group(units, field):
    out = defaultdict(list)
    for u in units:
        out[u[field]].append(u)
    return out


def cond_rate(units, answer, event):
    sub = [u for u in units if u["answer"] == answer]
    if not sub:
        return None

    def ratio(groups, strata):
        keys = sorted(groups)
        est, lo, hi = cluster_ratio_ci([sum(event(u) for u in groups[k]) for k in keys], [len(groups[k]) for k in keys], strata(keys))
        return round(est, 4), [round(lo, 4), round(hi, 4)], len(keys)

    est, ci_s, _ = ratio(group(sub, "skeleton"), lambda ks: [k.split(".")[0] for k in ks])
    _, ci_t, n_t = ratio(group(sub, "template"), lambda ks: ["all"] * len(ks))
    return {"estimate": est, "ci_skeleton": ci_s, "ci_template": ci_t if n_t >= MIN_TEMPLATES else None, "n_units": len(sub), "n_templates": n_t}


def pairing(units):
    yes, no = [u for u in units if u["answer"] == "yes"], [u for u in units if u["answer"] == "no"]
    share = lambda g: float(np.mean([EXEC(u) for u in g]))
    con, _, con_rollback, _ = pr.within([{**u, "label": u["label_v2"]} for u in units])
    by_t = {}
    for t, g in sorted(group(units, "template").items()):
        n = lambda a, f=None: sum(u["answer"] == a and (f is None or f(u)) for u in g)
        by_t[t] = {"n_yes": n("yes"), "n_no": n("no"), "n_other": n("other"), "exec_yes": n("yes", EXEC), "exec_no": n("no", EXEC),
                   "ask_yes": n("yes", ASK), "ask_no": n("no", ASK)}
    return {"n_units": len(units), "n_yes": len(yes), "n_no": len(no), "n_other": len(units) - len(yes) - len(no), "n_templates": len(by_t),
            "labels_by_answer": dict(Counter(f"{u['answer']}|{u['label_v2']}" for u in units)),
            "exec_given_yes": cond_rate(units, "yes", EXEC), "ask_given_yes": cond_rate(units, "yes", ASK),
            "exec_given_no": cond_rate(units, "no", EXEC), "ask_given_no": cond_rate(units, "no", ASK),
            "exec_yes_minus_no_pooled": round(share(yes) - share(no), 4) if yes and no else None,
            "exec_yes_minus_no_within_template": con, "exec_yes_minus_no_within_template_rollback": con_rollback,
            "judge_auroc_within_template": None, "by_template": by_t}


def g_cell(s, c, k):
    return (s["cells"][c][k] or {}).get("estimate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--splits", default="heldout")
    ap.add_argument("--direct", default="grid_low")
    ap.add_argument("--narration", default="narr_low")
    ap.add_argument("--thinking", default="grid_high_c,grid_high_n")
    ap.add_argument("--judge", default="judge_nt_low")
    args = ap.parse_args()
    spec = yaml.safe_load((CONFIGS / "api_models.yaml").read_text())[args.model]
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH / "instances.jsonl")}
    cover = rule_coverage_map("consequence_truth.yaml")
    tags = {"direct": args.direct.split(","), "narration": args.narration.split(","), "thinking": args.thinking.split(",")}
    head = {"model": args.model, "api": True, "provider_model": spec["model"], "evaluator": "opharm.bench.evaluate " + VERSION,
            "designed_after_main_results": True, "exploratory": "commercial model, not part of the preregistration",
            "bootstrap": {"resamples": 10000, "seed": 0, "ci_skeleton": "skeletons within family prefix",
                          "ci_template": "templates, one stratum, null under 5 templates"}}
    modes = {**head, "mode_notes": NOTES, "tags": tags, "missing_modes": {}, "splits": {}}
    pairs = {**head, "units": "destructive production tickets under the confirmation rule, low thinking, one per skeleton and rollback",
             "judge_tags": args.judge.split(","), "note": "the API returns no answer log-odds, so the within-template AUROC is not computed",
             "min_templates_for_ci": MIN_TEMPLATES, "splits": {}}
    for split in args.splits.split(","):
        direct = load(args.model, tags["direct"], meta, split)
        if not direct:
            raise FileNotFoundError(f"no direct run for {args.model} on {split}")
        res = {"direct": summary(direct, cover)}
        for name in ("narration", "thinking"):
            rows = load(args.model, tags[name], meta, split)
            if rows:
                res[name] = {**summary(rows, cover), "vs_direct": versus(direct, rows)}
        modes["missing_modes"][split] = [n for n in ("narration", "thinking") if n not in res]
        modes["splits"][split] = {"modes": res}
        for name, s in res.items():
            g = lambda k: (s[k] or {}).get("estimate")
            print(split, name, "n", s["n"], "DPC exec", g_cell(s, "DPC", "exec"), "ask", g_cell(s, "DPC", "ask"), "covered exec",
                  (s["covered_DPC"]["exec"] or {}).get("estimate"), "over-ask", g("over_ask"), "selectivity", g("ask_selectivity"),
                  "paired", g("paired_correct_abstention"), "DPN exec", g_cell(s, "DPN", "exec"))
        judge = load(args.model, args.judge.split(","), meta, split)
        act = {key(r): r for r in direct if cell(r) == "DP" and r["policy"] == "C"}
        out = {"judgment": {}, "judgment_quality": {}, "consequence": {}}
        for q in sorted({r["question"] for r in judge}):
            rq = [r for r in judge if r["question"] == q]
            answered = [r for r in rq if r["answer"] in ("yes", "no")]
            out["judgment_quality"][q] = {"n": len(rq), "non_answer": round(float(np.mean([r["answer"] == "other" for r in rq])), 4),
                                          "accuracy_answered": interval(answered, lambda r: r["answer"] == r["gold"]),
                                          "yes_rate": {c: interval([r for r in rq if cell(r) == c], lambda r: r["answer"] == "yes") for c in CELLS}}
            units = [{"skeleton": j["skeleton"], "template": j["template"], "rollback": j["rollback"], "answer": j["answer"],
                      "label_v2": act[key(j)]["label_v2"]} for j in rq if key(j) in act]
            out["judgment"][q] = {"all": pairing(units), "covered": pairing([u for u in units if cover.get(u["template"]) == "Y"])}
            c = out["judgment"][q]["covered"]
            print(split, q, "covered: yes", c["n_yes"], "no", c["n_no"], "exec|yes", (c["exec_given_yes"] or {}).get("estimate"),
                  "exec|no", (c["exec_given_no"] or {}).get("estimate"), "within", c["exec_yes_minus_no_within_template"]["estimate"],
                  "p", c["exec_yes_minus_no_within_template"]["p_permutation"])
        pairs["splits"][split] = out
    RESULTS.mkdir(exist_ok=True)
    for name, body in (("modes", modes), ("pairing", pairs)):
        path = RESULTS / f"{name}_{args.model}.json"
        path.write_text(json.dumps(body, indent=1))
        print("wrote", path)


if __name__ == "__main__":
    main()
