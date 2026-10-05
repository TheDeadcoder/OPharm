import argparse
import hashlib
import importlib
import itertools
import json
import sys
from collections import Counter, defaultdict

import opharm
import numpy as np
import yaml

from opharm.bench.evaluate import VERSION
from opharm.bench.labels import final_label, label, run_max_new
from opharm.paths import BENCH, CONFIGS, RESULTS, ROOT, RUNS

sys.path.insert(0, str(ROOT / "scripts"))
ti = importlib.import_module("28_template_intervals")

SAFE = {"ASK", "DECLINE"}
EVIDENCE = ("state", "health", "drill")
CONTRASTS = (("drill_minus_health", "drill", "health"), ("drill_minus_state", "drill", "state"), ("health_minus_state", "health", "state"))


def api_models():
    path = CONFIGS / "api_models.yaml"
    return set(yaml.safe_load(path.read_text())) if path.exists() else set()


def load(model, tag, meta, final):
    run = RUNS / model / tag
    cap = None if model in api_models() else run_max_new(model, tag)
    rows, seen = [], set()
    for name in ("results.jsonl", "samples.jsonl"):
        for r in map(json.loads, open(run / name)) if (run / name).exists() else ():
            k = (r["id"], r.get("sample", 0))
            if r["id"] not in meta or "text" not in r or k in seen:
                continue
            seen.add(k)
            row = {**meta[r["id"]], **r, "sample": k[1]}
            row["label_v2"] = (final_label(row, model, tag) if final else label(r["text"], row, model, cap))["label"]
            rows.append(row)
    return rows


def sign_flip(per_template):
    d = np.array(per_template)
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return round(float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()), 4)


def pairs(rows, policy, evidence):
    by = defaultdict(dict)
    for r in rows:
        if r["policy"] == policy and r["evidence"] == evidence:
            by[(r["skeleton"], r["sample"])][r["world"]] = r["label_v2"]
    return {k: d for k, d in by.items() if {"L", "R"} <= set(d)}


def gap(d):
    return float(d["L"] == "ASK") - float(d["R"] == "ASK")


def group_mean(p, name):
    by = defaultdict(list)
    for (s, _), d in p.items():
        by[name(s)].append(gap(d))
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())}


def examples(rows, k=3):
    out = defaultdict(list)
    for r in rows:
        if r["label_v2"] not in ("EXEC_MATCH", "ASK") and len(out[r["label_v2"]]) < k:
            out[r["label_v2"]].append({"id": r["id"], "reply_tail": r["text"][-300:]})
    return dict(out)


def summary(rows):
    out = {"n": len(rows), "n_samples": len({r["sample"] for r in rows}), "cells": {}, "labels": {}, "examples_other_labels": examples(rows),
           "selectivity": {}, "exec_gap": {}, "paired_correct": {}, "contrasts": {}, "by_mechanism": {}, "by_template": {}}
    mech = {r["skeleton"]: r["cls"] for r in rows}
    for policy in "RN":
        for evidence in EVIDENCE:
            for world in "LR":
                sub = [r for r in rows if (r["policy"], r["evidence"], r["world"]) == (policy, evidence, world)]
                key = f"{policy}.{evidence}.{world}"
                out["cells"][key] = {"exec": ti.interval(sub, ti.EXEC), "ask": ti.interval(sub, ti.ASK),
                                     "other": ti.interval(sub, lambda r: r["label_v2"] not in ("EXEC_MATCH", "ASK"))}
                out["labels"][key] = dict(Counter(r["label_v2"] for r in sub))
            p = pairs(rows, policy, evidence)
            items = [{"skeleton": s, "d": d} for (s, _), d in p.items()]
            key = f"{policy}.{evidence}"
            out["selectivity"][key] = ti.interval(items, lambda r: gap(r["d"]))
            out["exec_gap"][key] = ti.interval(items, lambda r: float(r["d"]["R"] == "EXEC_MATCH") - float(r["d"]["L"] == "EXEC_MATCH"))
            out["paired_correct"][key] = ti.interval(items, lambda r: r["d"]["L"] in SAFE and r["d"]["R"] == "EXEC_MATCH")
            out["by_mechanism"][key] = group_mean(p, lambda s: mech[s])
            out["by_template"][key] = group_mean(p, lambda s: s.rsplit(".", 1)[0])
        for name, a, b in CONTRASTS:
            pa, pb = pairs(rows, policy, a), pairs(rows, policy, b)
            items = [{"skeleton": k[0], "v": gap(pa[k]) - gap(pb[k])} for k in sorted(set(pa) & set(pb))]
            entry = ti.interval(items, lambda r: r["v"])
            if entry:
                by_t = defaultdict(list)
                for it in items:
                    by_t[it["skeleton"].rsplit(".", 1)[0]].append(it["v"])
                entry["p_sign_flip_templates"] = sign_flip([float(np.mean(v)) for _, v in sorted(by_t.items())])
            out["contrasts"][f"{policy}.{name}"] = entry
    if out["n_samples"] > 1:
        by = defaultdict(set)
        for r in rows:
            by[r["id"]].add(r["label_v2"])
        out["sample_agreement"] = round(float(np.mean([len(v) == 1 for v in by.values()])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", default="")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    api = args.model in api_models()
    runs = dict(x.split("=") for x in (args.runs or ("direct=rec_low,thinking=rec_high" if api else "direct=rec,thinking=rec_think")).split(","))
    fixtures = json.loads((RESULTS / "recovery_fixtures.json").read_text())
    sha = fixtures["sha256_instances"]
    if hashlib.sha256((BENCH / "recovery.jsonl").read_bytes()).hexdigest() != sha:
        raise ValueError("benchmark/recovery.jsonl differs from results/recovery_fixtures.json; rebuild it with 34_recovery_build.py")
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "recovery.jsonl"))}
    out = {"model": args.model, "api": api, "evaluator": "opharm.bench.evaluate " + VERSION,
           "sha256_instances": sha,
           "bootstrap": {"resamples": 10000, "seed": 0, "ci_skeleton": "skeletons, one stratum", "ci_template": "templates, one stratum",
                         "p_sign_flip_templates": "exact sign-flip test over template means, two-sided"},
           "missing_runs": [], "runs": {}}
    if args.final:
        out["labels"] = "evaluator v2 after continuation of capped replies"
    for name, tag in runs.items():
        run = RUNS / args.model / tag
        if not (run / "results.jsonl").exists():
            out["missing_runs"].append(name)
            continue
        used = json.loads((run / "manifest.json").read_text()).get("instances_sha256") if (run / "manifest.json").exists() else None
        if used and used != sha:
            raise ValueError(f"{args.model}/{tag} ran on a different instance file ({used[:12]})")
        res = out["runs"][name] = {"tag": tag, **summary(load(args.model, tag, meta, args.final))}
        for policy in "RN":
            g = lambda k, e: (res[k][f"{policy}.{e}"] or {}).get("estimate")
            c = res["contrasts"][f"{policy}.drill_minus_health"] or {}
            ask = lambda e, w: (res["cells"][f"{policy}.{e}.{w}"]["ask"] or {}).get("estimate")
            print(name, "rule" if policy == "R" else "neutral", "n", res["n"], "| ask L/R:", " ".join(f"{e} {ask(e, 'L')}/{ask(e, 'R')}" for e in EVIDENCE),
                  "| selectivity", [g("selectivity", e) for e in EVIDENCE], "| exec gap", [g("exec_gap", e) for e in EVIDENCE],
                  "| drill minus health", c.get("estimate"), c.get("ci_template"), "p", c.get("p_sign_flip_templates"))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"recovery_{args.model}{'_final' if args.final else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
