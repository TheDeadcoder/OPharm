import argparse
import importlib
import json
import sys
from collections import defaultdict

import opharm
import numpy as np

from opharm.analysis import code, rule_coverage_map
from opharm.bench.evaluate import VERSION
from opharm.paths import BENCH, RESULTS, ROOT
from opharm.stats.bootstrap import cluster_ci, template_ci

sys.path.insert(0, str(ROOT / "scripts"))
sec = importlib.import_module("18_secondary")

EXEC = lambda r: r["label_v2"] == "EXEC_MATCH"
ASK = lambda r: r["label_v2"] == "ASK"
MIN_TEMPLATES = 5


def interval(rows, value):
    if not rows:
        return None
    vals, skels = sec.per_skeleton(rows, value)
    est, lo, hi = cluster_ci(vals, [s.split(".")[0] for s in skels])
    templates = [s.rsplit(".", 1)[0] for s in skels]
    n_t = len(set(templates))
    ct = [round(v, 4) for v in template_ci(vals, templates)[1:]] if n_t >= MIN_TEMPLATES else None
    return {"estimate": round(est, 4), "ci_skeleton": [round(lo, 4), round(hi, 4)], "ci_template": ct,
            "n": len(rows), "n_skeletons": len(skels), "n_templates": n_t}


def selectivity(rows):
    by = defaultdict(dict)
    for r in rows:
        if r["policy"] == "C":
            by[(r["skeleton"], r["rollback"])][sec.cell(r)] = r["label_v2"]
    items = lambda a, b: [{"skeleton": s, "d": d} for (s, _), d in by.items() if {a, b} <= set(d)]
    return {"ask_env": interval(items("DP", "DS"), lambda r: float(r["d"]["DP"] == "ASK") - float(r["d"]["DS"] == "ASK")),
            "ask_target": interval(items("DP", "BP"), lambda r: float(r["d"]["DP"] == "ASK") - float(r["d"]["BP"] == "ASK")),
            "paired_correct_abstention": interval(items("DP", "DS"), lambda r: r["d"]["DP"] in sec.SAFE and r["d"]["DS"] == "EXEC_MATCH")}


def m_effects(rows):
    m = {(r["skeleton"], code(r)): r["m"] for r in rows}
    diff = lambda s, pairs: float(np.mean([m[(s, a)] - m[(s, b)] for a, b in pairs]))
    items = [{"skeleton": s, "policy": diff(s, [("DPAN", "DPAC"), ("DPNN", "DPNC")]),
              "blast": diff(s, [("DSAN", "DPAN"), ("DSNN", "DPNN")])} for s in sorted({r["skeleton"] for r in rows})]
    return {"delta_policy": interval(items, lambda r: r["policy"]), "delta_blast": interval(items, lambda r: r["blast"]),
            "policy_minus_blast": interval(items, lambda r: r["policy"] - r["blast"])}


def report(rows, cover):
    dpc = [r for r in rows if sec.cell(r) == "DP" and r["policy"] == "C"]
    cells = {}
    for p in "CN":
        for c in sec.CELLS:
            sub = [r for r in rows if sec.cell(r) == c and r["policy"] == p]
            cells[f"{c}{p}"] = {"exec": interval(sub, EXEC), "ask": interval(sub, ASK)}
    coverage = {}
    for g in ("Y", "?", "N"):
        sub = [r for r in dpc if cover.get(r["template"]) == g]
        coverage[g] = {"exec": interval(sub, EXEC), "ask": interval(sub, ASK)}
    return {"n_skeletons": len({r["skeleton"] for r in rows}), "n_templates": len({r["template"] for r in rows}),
            "cells": cells, "coverage": coverage, "selectivity": selectivity(rows), "m": m_effects(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--splits", default="dev,heldout,all")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH / "instances.jsonl")}
    cover = rule_coverage_map("consequence_truth.yaml")
    out = {"model": args.model, "evaluator": "opharm.bench.evaluate " + VERSION}
    if args.final:
        out["labels"] = "after continuation of capped replies"
    out.update(bootstrap={"resamples": 10000, "seed": 0, "ci_skeleton": "skeletons within family prefix (registered)",
                          "ci_template": "templates, one stratum, null under 5 templates"}, splits={})
    for split in args.splits.split(","):
        rows = [r for r in sec.load(args.model, "grid", meta, split, label=True, final=args.final) if r["set"] == "main"]
        out["splits"][split] = report(rows, cover)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"template_intervals_{args.model}{'_final' if args.final else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    for split, res in out["splits"].items():
        y = res["coverage"]["Y"]["exec"]
        print(f"{split}: coverage Y exec {y['estimate']} skeleton {y['ci_skeleton']} template {y['ci_template']}")


if __name__ == "__main__":
    main()
