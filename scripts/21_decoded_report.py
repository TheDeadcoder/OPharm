import argparse
import json
from collections import Counter, defaultdict

import opharm
import numpy as np

from opharm.bench.evaluate import VERSION
from opharm.bench.labels import label
from opharm.paths import BENCH, RESULTS, RUNS
from opharm.stats.bootstrap import cluster_ci

CELLS = ("DP", "DS", "BP", "BS")
BROKEN = {"MALFORMED", "TEXT_OTHER", "UNCERTAIN", "CLAIMED_DONE"}


def skel(i):
    return i.rsplit(".", 2)[0]


def ci(values, skels):
    return [round(v, 4) for v in cluster_ci(values, [s.split(".")[0] for s in skels])]


def paired(recs, base, cond, cells, policy, value):
    by = defaultdict(list)
    idx = {(r["id"], r["condition"]): r for r in recs}
    for r in recs:
        if r["condition"] != base or r["cell"][:2] not in cells or r["cell"][3] != policy:
            continue
        other = idx.get((r["id"], cond))
        if other:
            by[skel(r["id"])].append(value(r) - value(other))
    skels = sorted(by)
    return ci([float(np.mean(by[s])) for s in skels], skels) if skels else None


def rate(recs, cond, cells, policy, value):
    by = defaultdict(list)
    for r in recs:
        if r["condition"] == cond and r["cell"][:2] in cells and r["cell"][3] == policy:
            by[skel(r["id"])].append(value(r))
    skels = sorted(by)
    return ci([float(np.mean(by[s])) for s in skels], skels) if skels else None


def gain_report(recs):
    exec_ = lambda r: float(r["label_v2"] == "EXEC_MATCH")
    ask = lambda r: float(r["label_v2"] == "ASK")
    broken = lambda r: float(r["label_v2"] in BROKEN)
    out = {}
    for cond in sorted({r["condition"] for r in recs}):
        entry = {}
        for p in "CN":
            entry[p] = {
                "exec_DP": rate(recs, cond, ("DP",), p, exec_), "ask_DP": rate(recs, cond, ("DP",), p, ask),
                "exec_should_act": rate(recs, cond, ("DS", "BP", "BS"), p, exec_),
                "broken_all": rate(recs, cond, CELLS, p, broken),
                "mean_m": {c: round(float(np.mean([r["m"] for r in recs if r["condition"] == cond and r["cell"][:2] == c and r["cell"][3] == p])), 3)
                           for c in CELLS},
                "labels": dict(Counter(r["label_v2"] for r in recs if r["condition"] == cond and r["cell"][3] == p)),
            }
            if cond != "baseline":
                entry[p]["dangerous_reduction_DP"] = paired(recs, "baseline", cond, ("DP",), p, exec_)
                entry[p]["completion_loss_should_act"] = paired(recs, "baseline", cond, ("DS", "BP", "BS"), p, exec_)
        out[cond] = entry
    ident = [(r, o) for r in recs if r["condition"] == "baseline" for o in recs
             if o["id"] == r["id"] and o["condition"] == "identity"]
    out["identity_check"] = {"n": len(ident), "same_text": sum(r["text"] == o["text"] for r, o in ident),
                             "max_m_diff": max((abs(r["m"] - o["m"]) for r, o in ident), default=None)}
    return out


def c2_report(recs):
    out = {}
    clean = {r["id"]: r for r in recs if r["condition"] == "clean"}
    for cond in sorted({r["condition"] for r in recs} - {"clean"}):
        rows = [r for r in recs if r["condition"] == cond]
        base = lambda r: clean[r["id"]]
        crossed = [base(r)["m"] < 0 < r["m"] for r in rows]
        executed = [r["label_v2"] == "EXEC_MATCH" for r in rows]
        by = defaultdict(list)
        for r, e in zip(rows, executed):
            by[skel(r["id"])].append(float(e))
        skels = sorted(by)
        out[cond] = {"n": len(rows), "executes": ci([float(np.mean(by[s])) for s in skels], skels),
                     "m_crossed_zero": round(float(np.mean(crossed)), 4),
                     "crossed_and_executes": int(sum(c and e for c, e in zip(crossed, executed))),
                     "crossed_not_executes": int(sum(c and not e for c, e in zip(crossed, executed))),
                     "executes_not_crossed": int(sum(e and not c for c, e in zip(crossed, executed))),
                     "labels": dict(Counter(r["label_v2"] for r in rows))}
    out["clean_labels"] = dict(Counter(r["label_v2"] for r in clean.values()))
    return out


def steer_report(recs):
    out = {}
    for cond in sorted({r["condition"] for r in recs}):
        name = "random" if cond.startswith("rand") else cond
        key = name if cond == "clean" else f"{name.split('|')[0]}|{cond.split('|')[1]}" if "|" in cond else name
        for grp in ("DP", "DS", "BP"):
            rows = [r for r in recs if r["condition"] == cond and r["cell"][:2] == grp]
            e = out.setdefault(key, {}).setdefault(grp, Counter())
            e.update(r["label_v2"] for r in rows)
    return {k: {g: {"n": sum(c.values()), "exec": round(c["EXEC_MATCH"] / max(1, sum(c.values())), 4),
                    "ask": round(c["ASK"] / max(1, sum(c.values())), 4), "decline": round(c["DECLINE"] / max(1, sum(c.values())), 4),
                    "labels": dict(c)} for g, c in v.items()} for k, v in out.items()}


def load_records(run, mode, sfx, model):
    rows = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
    info = run / f"decoded_{mode}{sfx}_meta.json"
    max_new = json.loads(info.read_text())["args"]["max_new"] if info.exists() else None
    recs = [json.loads(line) for line in open(run / f"decoded_{mode}{sfx}.jsonl")]
    for r in recs:
        r["label_stored"] = r["label_v2"]
        r["label_v2"] = label(r["text"], rows[r["id"]], model, max_new)["label"]
    return recs


def compare(run, mode, a, b, model):
    load = lambda sfx: {(r["id"], r["condition"]): r for r in load_records(run, mode, sfx, model)}
    x, y = load(a), load(b)
    keys = sorted(set(x) & set(y))
    flips = Counter((x[k]["label_v2"], y[k]["label_v2"]) for k in keys if x[k]["label_v2"] != y[k]["label_v2"])
    return {"mode": mode, "a": a, "b": b, "n": len(keys), "only_a": len(set(x) - set(y)), "only_b": len(set(y) - set(x)),
            "label_agreement": round(sum(x[k]["label_v2"] == y[k]["label_v2"] for k in keys) / max(1, len(keys)), 4),
            "text_identical": round(sum(x[k]["text"] == y[k]["text"] for k in keys) / max(1, len(keys)), 4),
            "max_m_diff": round(max((abs(x[k]["m"] - y[k]["m"]) for k in keys), default=0.0), 4),
            "flips": {f"{p}->{q}": n for (p, q), n in flips.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--mode", default="steer")
    args = ap.parse_args()
    run = RUNS / args.model / args.tag
    if args.compare:
        a, b = args.compare.split(",")
        res = compare(run, args.mode, a, b, args.model)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"decoded_compare_{args.model}_{args.mode}{a}{b}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    sfx = args.suffix
    load = lambda mode: load_records(run, mode, sfx, args.model)
    report = {"model": args.model, "suffix": sfx, "evaluator": VERSION}
    if (run / f"decoded_gain{sfx}.jsonl").exists():
        report["gain"] = gain_report(load("gain"))
        report["gain_meta"] = json.loads((run / f"decoded_gain{sfx}_meta.json").read_text())
    if (run / f"decoded_c2{sfx}.jsonl").exists():
        report["c2"] = c2_report(load("c2"))
    if (run / f"decoded_steer{sfx}.jsonl").exists():
        report["steer"] = steer_report(load("steer"))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"decoded_{args.model}{sfx}.json").write_text(json.dumps(report, indent=1))
    print("wrote", RESULTS / f"decoded_{args.model}{sfx}.json")


if __name__ == "__main__":
    main()
