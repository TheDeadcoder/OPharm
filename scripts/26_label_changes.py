import argparse
import json
import subprocess
import types
from collections import Counter

import opharm

from opharm.bench.evaluate import VERSION
from opharm.bench.labels import label, run_max_new
from opharm.paths import BENCH, RESULTS, ROOT, RUNS
from opharm.stats.lock import analysis_rows

MODELS = ("qwen35_4b", "qwen35_9b", "qwen3_4b_2507", "gemma4_e4b", "llama31_8b")


def old_evaluator(commit):
    src = subprocess.run(["git", "show", f"{commit}:src/opharm/bench/evaluate.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("evaluate_old")
    exec(compile(src, f"evaluate@{commit}", "exec"), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="4982069")
    args = ap.parse_args()
    old = old_evaluator(args.old)
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
    out = {"old": f"{old.VERSION}@{args.old}", "new": VERSION, "models": {}}
    for model in MODELS:
        rows = analysis_rows([{**meta[r["id"]], **r} for r in map(json.loads, open(RUNS / model / "grid" / "results.jsonl"))
                              if r["id"] in meta and meta[r["id"]]["set"] in ("main", "shortcut") and "text" in r], model)
        max_new = run_max_new(model, "grid")
        changes, rates = Counter(), {}
        for r in rows:
            r["v20"] = old.evaluate(r["text"], r, model)["label"]
            r["v21"] = label(r["text"], r, model, max_new)["label"]
            r["strict"] = label(r["text"], r, model, max_new, strict=True)["label"]
            if r["v20"] != r["v21"]:
                changes[(r["split"], r["set"], r.get("target", "") + r["env"] + r["policy"], r["v20"], r["v21"])] += 1
        for split in ("dev", "heldout"):
            dpc = [r for r in rows if r["set"] == "main" and r["split"] == split and r["target"] + r["env"] + r["policy"] == "DPC"]
            rates[split] = {"n": len(dpc), **{f"{k}_{lab}": round(sum(r[k] == lab for r in dpc) / len(dpc), 4)
                                              for k in ("v20", "v21", "strict") for lab in ("EXEC_MATCH", "ASK", "INCOMPLETE")}}
        out["models"][model] = {"n": len(rows), "changed": sum(changes.values()), "dpc_rates": rates,
                                "changes": [{"split": s, "set": st, "cell": c, "v20": a, "v21": b, "n": n}
                                            for (s, st, c, a, b), n in sorted(changes.items())]}
        print(model, len(rows), "labels,", sum(changes.values()), "changed", json.dumps(rates["heldout"]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "label_changes_v20_v21.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
