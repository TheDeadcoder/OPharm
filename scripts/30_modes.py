import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict

import opharm
import numpy as np

from opharm.analysis import rule_coverage_map
from opharm.bench.evaluate import VERSION
from opharm.paths import BENCH, RESULTS, ROOT, RUNS

sys.path.insert(0, str(ROOT / "scripts"))
sec = importlib.import_module("18_secondary")
tiv = importlib.import_module("28_template_intervals")
interval, EXEC, ASK, selectivity = tiv.interval, tiv.EXEC, tiv.ASK, tiv.selectivity

OTHER = ("DS", "BP", "BS")


def ask_selectivity(rows):
    by = defaultdict(lambda: ([], []))
    for r in rows:
        if r["policy"] == "C":
            by[r["skeleton"]][sec.cell(r) != "DP"].append(ASK(r))
    items = [{"skeleton": s, "v": float(np.mean(dp)) - float(np.mean(rest))} for s, (dp, rest) in by.items() if dp and rest]
    return interval(items, lambda r: r["v"])


def mode_report(rows, cover):
    cells = {}
    for p in "CN":
        for c in sec.CELLS:
            sub = [r for r in rows if sec.cell(r) == c and r["policy"] == p]
            cells[f"{c}{p}"] = {"exec": interval(sub, EXEC), "ask": interval(sub, ASK), "labels": dict(sorted(Counter(r["label_v2"] for r in sub).items()))}
    covered = [r for r in rows if sec.cell(r) == "DP" and r["policy"] == "C" and cover.get(r["template"]) == "Y"]
    other = [r for r in rows if sec.cell(r) in OTHER and r["policy"] == "C"]
    sel = selectivity(rows)
    return {"n": len(rows), "cells": cells, "covered_DPC": {"exec": interval(covered, EXEC), "ask": interval(covered, ASK)},
            "over_ask": interval(other, ASK), "executes_where_allowed": interval(other, EXEC), "ask_selectivity": ask_selectivity(rows),
            "ask_env_selectivity": sel["ask_env"], "paired_correct_abstention": sel["paired_correct_abstention"]}


def vs_direct(rows, direct):
    d = {sec.key(r): r for r in direct}
    pairs = [(d[sec.key(r)], r) for r in rows if sec.key(r) in d]
    diff = lambda sub, f: interval([{"skeleton": a["skeleton"], "v": float(f(b)) - float(f(a))} for a, b in sub], lambda r: r["v"])
    res = {"n_matched": len(pairs), "covers_all_direct_prompts": len(pairs) == len(direct), "cells": {}}
    for p in "CN":
        for c in sec.CELLS:
            sub = [(a, b) for a, b in pairs if sec.cell(a) == c and a["policy"] == p]
            res["cells"][f"{c}{p}"] = {"exec_difference": diff(sub, EXEC), "ask_difference": diff(sub, ASK)}
    res["over_ask_difference"] = diff([(a, b) for a, b in pairs if sec.cell(a) in OTHER and a["policy"] == "C"], ASK)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--splits", default="dev,heldout")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if not (RUNS / args.model / "grid" / "results.jsonl").exists():
        raise FileNotFoundError(f"{args.model}: no grid run")
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH / "instances.jsonl")}
    cover = rule_coverage_map("consequence_truth.yaml")
    out = {"model": args.model, "evaluator": "opharm.bench.evaluate " + VERSION}
    if args.final:
        out["labels"] = "after continuation of capped replies"
    out.update(designed_after_main_results=True,
               bootstrap={"resamples": 10000, "seed": 0, "ci_skeleton": "skeletons within family prefix (registered)",
                          "ci_template": "templates, one stratum, null under 5 templates"}, missing_modes={}, splits={})
    for split in args.splits.split(","):
        direct = [r for r in sec.load(args.model, "grid", meta, split, label=True, final=args.final) if r["set"] == "main"]
        loaded = {"direct": direct,
                  "narration": sec.load(args.model, "narr", meta, split, label=True, final=args.final)
                  if (RUNS / args.model / "narr" / "results.jsonl").exists() else [],
                  "thinking": sec.load_tags(args.model, ("think", "think_rest"), meta, split, label=True, final=args.final)}
        out["missing_modes"][split] = [m for m, rows in loaded.items() if not rows]
        modes = {}
        for m, rows in loaded.items():
            if rows:
                modes[m] = mode_report(rows, cover)
                if m != "direct":
                    modes[m]["vs_direct"] = vs_direct(rows, direct)
        out["splits"][split] = {"modes": modes}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"modes_{args.model}{'_final' if args.final else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    est = lambda e: None if e is None else e["estimate"]
    for split, res in out["splits"].items():
        for m, v in res["modes"].items():
            print(f"{split} {m}: n {v['n']} DPC exec {est(v['cells']['DPC']['exec'])} ask {est(v['cells']['DPC']['ask'])} "
                  f"covered exec {est(v['covered_DPC']['exec'])} over_ask {est(v['over_ask'])} ask_selectivity {est(v['ask_selectivity'])} "
                  f"paired_correct_abstention {est(v['paired_correct_abstention'])} DPN exec {est(v['cells']['DPN']['exec'])}")


if __name__ == "__main__":
    main()
