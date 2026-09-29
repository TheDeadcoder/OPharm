import argparse
import json
from collections import Counter

import opharm

from opharm.bench.evaluate import VERSION
from opharm.bench.labels import label, run_max_new
from opharm.paths import BENCH, RESULTS, RUNS
from opharm.stats.lock import analysis_rows, dev_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tags", default="grid")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
    sfx = "_confirm" if args.confirm else "_dev"
    summary = {"model": args.model, "evaluator": VERSION, "runs": {}}
    for tag in args.tags.split(","):
        rows = [dict(meta[r["id"]], **r) for r in map(json.loads, open(RUNS / args.model / tag / "results.jsonl"))
                if r["id"] in meta and "text" in r]
        rows = analysis_rows(rows, args.model) if args.confirm else dev_only(rows)
        out, table = [], Counter()
        for r in rows:
            v2 = label(r["text"], r, args.model, run_max_new(args.model, tag))
            out.append({"id": r["id"], "label_v1": r["label"], "label_v2": v2["label"],
                        "label_strict": label(r["text"], r, args.model, run_max_new(args.model, tag), strict=True)["label"],
                        **{k: v2[k] for k in ("reason", "ask_via_tool", "protocol", "label_untruncated") if k in v2}})
            table[(r["split"], r["set"], r["label"], v2["label"])] += 1
        (RUNS / args.model / tag / f"labels_v2{sfx}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
        summary["runs"][tag] = {"n": len(out), "v1_to_v2": [{"split": s, "set": st, "v1": a, "v2": b, "n": n}
                                                            for (s, st, a, b), n in sorted(table.items())]}
        print(f"{args.model} {tag}: {len(out)} outputs, {sum(n for (_, _, a, b), n in table.items() if a != b)} relabeled")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"labels_v2_{args.model}{sfx}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
