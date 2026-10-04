import argparse
import importlib
import json
import sys
from collections import defaultdict

import opharm

from opharm.analysis import rule_coverage_map
from opharm.paths import BENCH, RESULTS, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
sec = importlib.import_module("18_secondary")
ti = importlib.import_module("28_template_intervals")


def rule_effect(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["skeleton"], sec.cell(r), r["rollback"])][r["policy"]] = r["label_v2"]
    out = {}
    for c in sec.CELLS:
        items = [{"skeleton": s, "d": d} for (s, cc, _), d in by.items() if cc == c and {"C", "N"} <= set(d)]
        out[c] = {"ask_C_minus_N": ti.interval(items, lambda r: float(r["d"]["C"] == "ASK") - float(r["d"]["N"] == "ASK")),
                  "exec_C_minus_N": ti.interval(items, lambda r: float(r["d"]["C"] == "EXEC_MATCH") - float(r["d"]["N"] == "EXEC_MATCH"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="main_dev")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    meta = {json.loads(line)["id"]: json.loads(line) for line in open(BENCH / "instances.jsonl")}
    rows = [r for r in sec.load(args.model, args.tag, meta, args.split, label=True, final=True) if r["set"] == "main"]
    out = {"model": args.model, "tag": args.tag, "split": args.split, "labels": "after continuation of capped replies",
           **ti.report(rows, rule_coverage_map("consequence_truth.yaml")), "rule_effect": rule_effect(rows)}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"scale_{args.model}_{args.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
