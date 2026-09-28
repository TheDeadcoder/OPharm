import argparse
import csv
import json
import random
from collections import Counter, defaultdict

import opharm

from opharm.bench.evaluate import evaluate
from opharm.paths import BENCH, RUNS
from opharm.stats.lock import analysis_rows, dev_only

QUOTA = {"EXEC_MATCH": 10, "ASK": 10, "INSPECT": 5, "EXEC_OTHER": 5, "EXEC_SAFE": 4, "EXEC_UNSAFE": 3, "DECLINE": 3,
         "TEXT_OTHER": 3, "MALFORMED": 2, "CLAIMED_DONE": 2, "NOOP": 2, "EXEC_OTHER_TARGET": 2, "UNCERTAIN": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen35_4b,qwen35_9b,qwen3_4b_2507")
    ap.add_argument("--heldout", default="qwen35_4b")
    ap.add_argument("--disagree", type=int, default=8)
    ap.add_argument("--random", type=int, default=15)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
    rng, items = random.Random(args.seed), []
    for model in args.models.split(","):
        rows = [dict(meta[r["id"]], **r) for r in map(json.loads, open(RUNS / model / "grid" / "results.jsonl"))
                if r["id"] in meta and meta[r["id"]]["set"] in ("main", "shortcut") and "text" in r]
        rows = analysis_rows(rows, model) if model in args.heldout.split(",") else dev_only(rows)
        pool, allrows = defaultdict(list), []
        for r in rows:
            v2 = evaluate(r["text"], r, model)
            rec = {"model": model, "instance": r["id"], "v1": r["label"], "v2": v2["label"], "via_tool": v2.get("ask_via_tool", False),
                   "ticket": "".join(t for _, t in r["user"]), "output": r["text"]}
            allrows.append(rec)
            pool["DISAGREE" if r["label"] != v2["label"] else v2["label"]].append(rec)
        picked = rng.sample(pool["DISAGREE"], min(args.disagree, len(pool["DISAGREE"])))
        for lab, n in QUOTA.items():
            cand = [x for x in pool[lab] if x not in picked]
            picked += rng.sample(cand, min(n, len(cand)))
        rest = [x for x in allrows if x not in picked]
        items += [dict(x, stratum="stratified") for x in picked] + [dict(x, stratum="random") for x in rng.sample(rest, args.random)]
    rng.shuffle(items)
    out = RUNS / "audit"
    out.mkdir(exist_ok=True)
    with open(out / "audit_items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "ticket", "model_output", "label_annotator", "notes"])
        for i, x in enumerate(items):
            w.writerow([f"A{i:03d}", x["ticket"], x["output"], "", ""])
    with open(out / "audit_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "model", "instance", "stratum", "label_v1", "label_v2", "ask_via_tool"])
        for i, x in enumerate(items):
            w.writerow([f"A{i:03d}", x["model"], x["instance"], x["stratum"], x["v1"], x["v2"], x["via_tool"]])
    print(len(items), "items", dict(Counter(x["model"] for x in items)))


if __name__ == "__main__":
    main()
