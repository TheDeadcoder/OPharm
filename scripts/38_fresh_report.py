import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter, defaultdict

import opharm
import numpy as np
import yaml

from opharm.analysis import rule_coverage_map
from opharm.bench.evaluate import VERSION
from opharm.bench.labels import final_label, label, run_max_new
from opharm.paths import BENCH, CONFIGS, RESULTS, ROOT, RUNS
from opharm.stats.bootstrap import cluster_ratio_ci

sys.path.insert(0, str(ROOT / "scripts"))
ti = importlib.import_module("28_template_intervals")

CELLS, OTHER_CELLS, MIN_TEMPLATES = ("DP", "DS", "BP", "BS"), ("DS", "BP", "BS"), 5
TRUTH, HARMS = "fresh_truth.yaml", (("data", "QD"), ("access", "QA"), ("service", "QO"))
LOCAL = "direct=fresh+fresh_rest,narration=fresh_narr+fresh_narr_rest,thinking=fresh_think+fresh_think_rest,judge=fresh_judge_nt+fresh_judge_nt_rest"
API = "direct=fresh_low,narration=fresh_narr_low,thinking=fresh_high,judge=fresh_judge_nt_low"
EXEC, ASK, interval = ti.EXEC, ti.ASK, ti.interval


def api_models():
    path = CONFIGS / "api_models.yaml"
    return set(yaml.safe_load(path.read_text())) if path.exists() else set()


def cell(r):
    return r["target"] + r["env"]


def harm(marks):
    d = lambda q: marks[q]["D"] if isinstance(marks[q], dict) else marks[q]
    return next((name for name, q in HARMS if d(q) == "Y"), "unclear")


def key(r):
    return r["skeleton"], cell(r) + r["rollback"] + r["policy"]


def load(model, tags, meta, sha, final):
    rows = []
    for tag in tags:
        run = RUNS / model / tag
        if not (run / "results.jsonl").exists():
            continue
        used = json.loads((run / "manifest.json").read_text()).get("instances_sha256") if (run / "manifest.json").exists() else None
        if used and used != sha:
            raise ValueError(f"{model}/{tag} ran on a different instance file ({used[:12]})")
        cap = None if model in api_models() else run_max_new(model, tag)
        for r in map(json.loads, open(run / "results.jsonl")):
            if r["id"] not in meta:
                continue
            row = {**meta[r["id"]], **r}
            if row["set"] != "judge":
                if "text" not in r:
                    continue
                row["label_v2"] = (final_label(row, model, tag) if final else label(r["text"], row, model, cap))["label"]
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{model} {tags}: the same instance appears in more than one run")
    return rows


def examples(rows, k=3):
    out = defaultdict(list)
    for r in rows:
        if r["label_v2"] not in ("EXEC_MATCH", "ASK") and len(out[r["label_v2"]]) < k:
            out[r["label_v2"]].append({"id": r["id"], "reply_tail": r["text"][-300:]})
    return dict(out)


def summary(rows, cover, harms):
    pick = lambda c, p: [r for r in rows if cell(r) == c and r["policy"] == p]
    free = [r for r in rows if r["policy"] == "C" and cell(r) in OTHER_CELLS]
    covered = [r for r in pick("DP", "C") if cover.get(r["template"]) == "Y"]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["policy"] == "C":
            by[r["skeleton"]]["need" if cell(r) == "DP" else "free"].append(float(ASK(r)))
    gaps = [{"skeleton": s, "v": float(np.mean(d["need"]) - np.mean(d["free"]))} for s, d in by.items() if d["need"] and d["free"]]
    sel = ti.selectivity(rows)
    per = defaultdict(list)
    for r in pick("DP", "C"):
        per[r["template"]].append(r["label_v2"])
    return {"n": len(rows), "n_skeletons": len({r["skeleton"] for r in rows}), "n_templates": len({r["template"] for r in rows}),
            "examples_other_labels": examples(rows),
            "cells": {c + p: {"exec": interval(pick(c, p), EXEC), "ask": interval(pick(c, p), ASK),
                              "labels": dict(Counter(r["label_v2"] for r in pick(c, p)))} for p in "CN" for c in CELLS},
            "covered_DPC": {"exec": interval(covered, EXEC), "ask": interval(covered, ASK)},
            "over_ask": interval(free, ASK), "executes_where_allowed": interval(free, EXEC),
            "ask_selectivity": interval(gaps, lambda r: r["v"]),
            "ask_env_selectivity": sel["ask_env"], "paired_correct_abstention": sel["paired_correct_abstention"],
            "DPC_by_harm": {h: {"exec": interval(sub, EXEC), "ask": interval(sub, ASK)} for h in ("data", "access", "service", "unclear")
                            for sub in [[r for r in pick("DP", "C") if harms[r["template"]] == h]] if sub},
            "DPC_by_template": {t: {"n": len(v), "exec": round(float(np.mean([x == "EXEC_MATCH" for x in v])), 4),
                                    "ask": round(float(np.mean([x == "ASK" for x in v])), 4), "rule_covers": cover.get(t), "harm": harms[t]}
                                for t, v in sorted(per.items())}}


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


def cond_rate(units, answer, event):
    sub = [u for u in units if u["answer"] == answer]
    if not sub:
        return None

    def ratio(field, strata):
        groups = defaultdict(list)
        for u in sub:
            groups[u[field]].append(u)
        keys = sorted(groups)
        est, lo, hi = cluster_ratio_ci([sum(event(u) for u in groups[k]) for k in keys], [len(groups[k]) for k in keys], strata(keys))
        return round(est, 4), [round(lo, 4), round(hi, 4)], len(keys)

    est, ci_s, _ = ratio("skeleton", lambda ks: [k.split(".")[0] for k in ks])
    _, ci_t, n_t = ratio("template", lambda ks: ["all"] * len(ks))
    return {"estimate": est, "ci_skeleton": ci_s, "ci_template": ci_t if n_t >= MIN_TEMPLATES else None, "n_units": len(sub), "n_templates": n_t}


def judgment(judge, direct, cover, q3_exclude):
    act = {key(r): r for r in direct if cell(r) == "DP" and r["policy"] == "C"}
    out = {}
    for q in sorted({r["question"] for r in judge}):
        rq = [r for r in judge if r["question"] == q and r["policy"] == "C" and not (q == "q3" and r["template"] in q3_exclude)]
        answered = [r for r in rq if r["answer"] in ("yes", "no")]
        units = [{"skeleton": j["skeleton"], "template": j["template"], "answer": j["answer"], "label_v2": act[key(j)]["label_v2"]} for j in rq if key(j) in act]
        scopes = {"all": units, "covered": [u for u in units if cover.get(u["template"]) == "Y"]}
        out[q] = {"n": len(rq), "non_answer": round(float(np.mean([r["answer"] == "other" for r in rq])), 4) if rq else None,
                  "accuracy_answered": interval(answered, lambda r: r["answer"] == r["gold"]),
                  "yes_rate": {c: interval([r for r in rq if cell(r) == c], lambda r: r["answer"] == "yes") for c in CELLS},
                  "paired_with_action": {name: {"n_units": len(u), "labels_by_answer": dict(Counter(f"{x['answer']}|{x['label_v2']}" for x in u)),
                                                "exec_given_yes": cond_rate(u, "yes", EXEC), "exec_given_no": cond_rate(u, "no", EXEC),
                                                "ask_given_yes": cond_rate(u, "yes", ASK), "ask_given_no": cond_rate(u, "no", ASK)}
                                         for name, u in scopes.items()}}
    return out


def reference(model, final):
    for stem in (f"modes_{model}_final", f"modes_{model}")[::1 if final else -1]:
        path = RESULTS / f"{stem}.json"
        if path.exists():
            modes = json.loads(path.read_text()).get("splits", {}).get("heldout", {}).get("modes", {})
            return {"source": stem, "modes": {m: {"DPC_exec": s["cells"]["DPC"]["exec"], "DPC_ask": s["cells"]["DPC"]["ask"],
                                                   "covered_DPC_exec": s["covered_DPC"]["exec"], "DPN_exec": s["cells"]["DPN"]["exec"],
                                                   "over_ask": s["over_ask"], "ask_selectivity": s["ask_selectivity"]} for m, s in modes.items()}}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", default="")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    api = args.model in api_models()
    if api and args.final:
        raise SystemExit("--final applies to local models only")
    tags = {k: v.split("+") for k, v in (x.split("=") for x in (args.runs or (API if api else LOCAL)).split(","))}
    info = json.loads((RESULTS / "fresh_templates.json").read_text())
    sha = info["sha256_instances"]
    if hashlib.sha256((BENCH / "fresh.jsonl").read_bytes()).hexdigest() != sha:
        raise ValueError("benchmark/fresh.jsonl differs from results/fresh_templates.json; rebuild it with 37_fresh_build.py")
    truth_bytes = (CONFIGS / TRUTH).read_bytes()
    if hashlib.sha256(truth_bytes).hexdigest() != info["truth"]["sha256"]:
        raise ValueError(f"configs/{TRUTH} changed after results/fresh_templates.json was written; rerun 37_fresh_build.py")
    truth = yaml.safe_load(truth_bytes)
    signed = str(truth.get("confirmed_by") or "").strip()
    print("consequence marks:", f"confirmed by {signed}" if signed else f"NOT confirmed (confirmed_by is empty in configs/{TRUTH})")
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "fresh.jsonl"))}
    cover, harms = rule_coverage_map(TRUTH), {t: harm(m) for t, m in truth["templates"].items()}
    q3_exclude = set(truth.get("q3_exclude") or [])
    direct = load(args.model, tags["direct"], meta, sha, args.final)
    if not direct:
        raise FileNotFoundError(f"no direct run of the fresh set for {args.model}")
    res = {"direct": summary(direct, cover, harms)}
    for name in ("narration", "thinking"):
        rows = load(args.model, tags.get(name, []), meta, sha, args.final)
        if rows:
            res[name] = {**summary(rows, cover, harms), "vs_direct": versus(direct, rows)}
    judge = load(args.model, tags.get("judge", []), meta, sha, args.final)
    found = {k: f for k, v in tags.items() if (f := [t for t in v if (RUNS / args.model / t / "results.jsonl").exists()])}
    out = {"model": args.model, "api": api, "evaluator": "opharm.bench.evaluate " + VERSION,
           "sha256_instances": sha,
           "truth": {**info["truth"], "confirmed_by": signed},
           "tags": found, "bootstrap": {"resamples": 10000, "seed": 0, "ci_skeleton": "skeletons within family prefix",
                                       "ci_template": "templates, one stratum, null under 5 templates"},
           "missing_modes": [n for n in ("narration", "thinking") if n not in res] + ([] if judge else ["judge"]), "modes": res,
           "judgment": judgment(judge, direct, cover, q3_exclude) if judge else None, "q3_excluded_templates": sorted(q3_exclude),
           "heldout_reference": reference(args.model, args.final)}
    if args.final:
        out["labels"] = "evaluator v2 after continuation of capped replies"
    g = lambda entry: (entry or {}).get("estimate")
    for name, s in res.items():
        print(name, "n", s["n"], "skeletons", s["n_skeletons"], "| DPC exec", g(s["cells"]["DPC"]["exec"]), "ask", g(s["cells"]["DPC"]["ask"]),
              "| covered exec", g(s["covered_DPC"]["exec"]), "| over-ask", g(s["over_ask"]), "| selectivity", g(s["ask_selectivity"]),
              "template", (s["ask_selectivity"] or {}).get("ci_template"), "| DPN exec", g(s["cells"]["DPN"]["exec"]))
        ref = (out["heldout_reference"] or {}).get("modes", {}).get(name)
        if ref:
            print("  held-out:", "DPC exec", g(ref["DPC_exec"]), "ask", g(ref["DPC_ask"]), "| covered exec", g(ref["covered_DPC_exec"]),
                  "| over-ask", g(ref["over_ask"]), "| selectivity", g(ref["ask_selectivity"]), "| DPN exec", g(ref["DPN_exec"]))
    for q, s in (out["judgment"] or {}).items():
        c = s["paired_with_action"]["covered"]
        print(q, "n", s["n"], "yes on DP", g(s["yes_rate"]["DP"]), "BS", g(s["yes_rate"]["BS"]), "| covered units", c["n_units"],
              "exec|yes", g(c["exec_given_yes"]), "exec|no", g(c["exec_given_no"]))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"fresh_{args.model}{'_final' if args.final else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
