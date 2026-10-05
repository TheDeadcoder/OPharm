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
from opharm.paths import BENCH, RESULTS, ROOT, RUNS
from opharm.stats.bootstrap import cluster_ratio_ci

sys.path.insert(0, str(ROOT / "scripts"))
sec = importlib.import_module("18_secondary")
cons = importlib.import_module("22_consequence")

B, MIN_TEMPLATES = 10000, 5
JUDGE_TAGS = ("judge_nt_pilot", "judge_nt")
CONS_TAGS = ["cons1", "cons2dp", "cons2", "consqp1", "consqp2dp", "consqp2"]
CONS_FILES = ["consequence_1.jsonl", "consequence_2.jsonl", "consequence_qp_1.jsonl", "consequence_qp_2.jsonl"]
NOTE = "judgment under the neutral rule, action under the confirmation rule"
EXEC = lambda u: u["label"] == "EXEC_MATCH"
ASK = lambda u: u["label"] == "ASK"
r4 = lambda v: round(float(v), 4)


def ratio(units, hit):
    if not units:
        return None
    out = {"estimate": r4(np.mean([hit(u) for u in units]))}
    for name, group, stratum in (("ci_skeleton", "skeleton", lambda k: k.split(".")[0]), ("ci_template", "template", lambda k: "all")):
        by = defaultdict(lambda: [0.0, 0.0])
        for u in units:
            by[u[group]][0] += hit(u)
            by[u[group]][1] += 1
        keys = sorted(by)
        out[name] = None if name == "ci_template" and len(keys) < MIN_TEMPLATES else \
            [r4(v) for v in cluster_ratio_ci([by[k][0] for k in keys], [by[k][1] for k in keys], [stratum(k) for k in keys])[1:]]
    return {**out, "n_units": len(units), "n_templates": len({u["template"] for u in units})}


def summary(nums, dens, null=None):
    if not dens:
        return {"estimate": None, "ci_template": None, "n_informative_templates": 0, "p_permutation": None}
    nums, dens = np.array(nums), np.array(dens, dtype=float)
    stat = nums.sum(0) / dens.sum()
    center = stat[1:].mean() if null is None else null
    far = np.abs(stat[1:] - center) >= abs(stat[0] - center) - 1e-12
    ci = [r4(v) for v in cluster_ratio_ci(nums[:, 0], dens, ["all"] * len(dens))[1:]] if len(dens) >= MIN_TEMPLATES else None
    out = {"estimate": r4(stat[0]), "ci_template": ci, "n_informative_templates": len(dens)}
    if null is None:
        out["permutation_mean"] = r4(center)
    return {**out, "p_permutation": r4((1 + far.sum()) / (B + 1))}


def within(units):
    by = defaultdict(lambda: defaultdict(dict))
    for u in units:
        by[u["template"]][u["skeleton"]][u["rollback"]] = u
    rng, scored = np.random.default_rng(0), all("judge" in u for u in units)
    acc = {k: ([], []) for k in ("contrast", "auroc", "contrast_rollback", "auroc_rollback")}
    for t in sorted(by):
        skels = sorted(by[t])
        if any(set(by[t][s]) != {"A", "N"} for s in skels):
            raise ValueError(f"{t}: a skeleton lacks one of its two rollback units")
        grid = [[by[t][s][rb] for rb in "AN"] for s in skels]
        arr = lambda f: np.array([[f(u) for u in row] for row in grid], dtype=float)
        ex, held, yes, no, judge = arr(EXEC), arr(lambda u: u["label"] in sec.SAFE), arr(lambda u: u["answer"] == "yes"), \
            arr(lambda u: u["answer"] == "no"), arr(lambda u: u["judge"] if scored else 0.0)
        perm = np.vstack([np.arange(len(skels)), np.argsort(rng.random((B, len(skels))), axis=1)])
        for name, cols in (("", [slice(0, 2)]), ("_rollback", [slice(0, 1), slice(1, 2)])):
            con, auc = [0.0, 0.0], [0.0, 0]
            for c in cols:
                e, y, n = ex[:, c], yes[:, c], no[:, c]
                ny, nn = y.sum(), n.sum()
                if ny and nn:
                    w = ny * nn / (ny + nn)
                    con = [con[0] + w * ((y[perm] * e).sum((1, 2)) / ny - (n[perm] * e).sum((1, 2)) / nn), con[1] + w]
                hi, xi = np.flatnonzero(held[:, c].ravel()), np.flatnonzero(e.ravel())
                if scored and len(hi) and len(xi):
                    jp = judge[:, c][perm].reshape(len(perm), -1)
                    a, b = jp[:, hi][:, :, None], jp[:, xi][:, None, :]
                    auc = [auc[0] + ((a > b) + 0.5 * (a == b)).sum((1, 2)), auc[1] + len(hi) * len(xi)]
            for key, (num, den) in (("contrast" + name, con), ("auroc" + name, auc)):
                if den:
                    acc[key][0].append(num)
                    acc[key][1].append(den)
    return (summary(*acc["contrast"]), summary(*acc["auroc"]), summary(*acc["contrast_rollback"], 0.0),
            summary(*acc["auroc_rollback"], 0.5))


def stats(units):
    yes, no = [u for u in units if u["answer"] == "yes"], [u for u in units if u["answer"] == "no"]
    res = {"n_units": len(units), "n_yes": len(yes), "n_no": len(no), "n_other": len(units) - len(yes) - len(no),
           "n_templates": len({u["template"] for u in units}),
           "labels_by_answer": dict(sorted(Counter(f"{u['answer']}|{u['label']}" for u in units).items())),
           "exec_given_yes": ratio(yes, EXEC), "ask_given_yes": ratio(yes, ASK), "exec_given_no": ratio(no, EXEC), "ask_given_no": ratio(no, ASK),
           "exec_yes_minus_no_pooled": r4(np.mean([EXEC(u) for u in yes]) - np.mean([EXEC(u) for u in no])) if yes and no else None}
    (res["exec_yes_minus_no_within_template"], res["judge_auroc_within_template"],
     res["exec_yes_minus_no_within_template_rollback"], res["judge_auroc_within_template_rollback"]) = within(units)
    count = lambda sub, f: sum(map(f, sub))
    res["by_template"] = {}
    for t in sorted({u["template"] for u in units}):
        ty, tn = [u for u in yes if u["template"] == t], [u for u in no if u["template"] == t]
        res["by_template"][t] = {"n_yes": len(ty), "n_no": len(tn), "n_other": sum(u["template"] == t for u in units) - len(ty) - len(tn),
                                 "exec_yes": count(ty, EXEC), "exec_no": count(tn, EXEC), "ask_yes": count(ty, ASK), "ask_no": count(tn, ASK)}
    return res


def unit(action, judged):
    return {"skeleton": action["skeleton"], "template": action["template"], "rollback": action["rollback"],
            "answer": judged["answer"], "judge": judged["judge"], "label": action["label_v2"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--splits", default="dev,heldout,all")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if not any((RUNS / args.model / t / "results.jsonl").exists() for t in JUDGE_TAGS):
        raise FileNotFoundError(f"{args.model}: no tool-free judgment run ({', '.join(JUDGE_TAGS)})")
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH / "instances.jsonl")}
    cover = rule_coverage_map("consequence_truth.yaml")
    truth = yaml.safe_load(cons.TRUTH.read_text())["templates"]
    out = {"model": args.model, "evaluator": "opharm.bench.evaluate " + VERSION}
    if args.final:
        out["labels"] = "after continuation of capped replies"
    out.update(units="destructive production tickets under the confirmation rule, direct mode, one per skeleton and rollback",
               bootstrap={"resamples": 10000, "seed": 0, "ci_skeleton": "skeletons within family prefix", "ci_template": "templates, one stratum",
                          "p_permutation": "10000 permutations of skeletons within template, both rollbacks together, two-sided: "
                                           "around the permutation mean for the statistics pooled over rollback, around 0 or 0.5 "
                                           "for the statistics within template and rollback"},
               min_templates_for_ci=MIN_TEMPLATES, splits={})
    for split in args.splits.split(","):
        acts = [r for r in sec.load(args.model, "grid", meta, split, label=True, final=args.final)
                if r["set"] == "main" and sec.cell(r) == "DP" and r["policy"] == "C"]
        by_key, by_rb = {sec.key(a): a for a in acts}, {(a["skeleton"], a["rollback"]): a for a in acts}
        judge = [j for j in sec.load_tags(args.model, JUDGE_TAGS, meta, split) if sec.key(j) in by_key]
        res = {"judgment": {}, "consequence": {}}
        for q in sorted({j["question"] for j in judge}):
            units = [unit(by_key[sec.key(j)], j) for j in judge if j["question"] == q]
            res["judgment"][q] = {"all": stats(units), "covered": stats([u for u in units if cover.get(u["template"]) == "Y"])}
        rows = cons.load(args.model, CONS_TAGS, CONS_FILES, {"dev", "heldout"} if split == "all" else {split})[0]
        for q in cons.OPERATION + cons.SUPPLEMENT:
            units = [unit(by_rb[(r["skeleton"], r["rollback"])], r) for r in rows
                     if r["question"] == q and r["variant"] == "base" and r["target"] == "D" and r["env"] == "P"
                     and cons.mark(truth, r["template"], q, "D") == "Y"]
            if units:
                res["consequence"][q] = {"note": NOTE, **stats(units)}
        out["splits"][split] = res
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"pairing_{args.model}{'_final' if args.final else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    est = lambda e: None if e is None else e["estimate"]
    for split, res in out["splits"].items():
        for q, v in res["judgment"].items():
            c = v["covered"]
            pr = lambda k: f"{c[k]['estimate']} (p {c[k]['p_permutation']})"
            print(f"{split} {q} covered: yes {c['n_yes']} no {c['n_no']} exec|yes {est(c['exec_given_yes'])} exec|no {est(c['exec_given_no'])} "
                  f"within {pr('exec_yes_minus_no_within_template')} auroc {pr('judge_auroc_within_template')} "
                  f"within rollback {pr('exec_yes_minus_no_within_template_rollback')} auroc {pr('judge_auroc_within_template_rollback')}")


if __name__ == "__main__":
    main()
