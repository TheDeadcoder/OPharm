import argparse
import json
import sys
from collections import Counter

import opharm

from opharm.bench.labels import label, run_max_new
from opharm.paths import BENCH, RESULTS, RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--instances", default="instances.jsonl")
    ap.add_argument("--min-parse", type=float, default=0.90)
    args = ap.parse_args()
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / args.instances))}
    budget = run_max_new(args.model, args.tag)
    labs, protocols, call_like, parsed = Counter(), Counter(), 0, 0
    for r in map(json.loads, open(RUNS / args.model / args.tag / "results.jsonl")):
        if "text" not in r:
            continue
        lab = label(r["text"], meta[r["id"]], args.model, budget)
        labs[lab["label"]] += 1
        protocols[lab["protocol"]] += 1
        if '{"name"' in r["text"] or lab["protocol"] in ("json", "tagged"):
            call_like += 1
            parsed += bool(lab["calls"]) and lab["label"] != "MALFORMED"
    rate = parsed / call_like if call_like else None
    out = {"model": args.model, "tag": args.tag, "n": sum(labs.values()), "labels": dict(labs), "protocols": dict(protocols),
           "call_like": call_like, "parsed": parsed, "parse_rate": None if rate is None else round(rate, 4), "min_parse": args.min_parse,
           "passed": rate is not None and rate >= args.min_parse}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"parse_check_{args.model}_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
