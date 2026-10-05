import argparse
import csv
import json
import random
import subprocess
from collections import Counter

import opharm

from opharm.bench.evaluate import VERSION
from opharm.bench.labels import capped, label, run_max_new
from opharm.paths import BENCH, ROOT, RUNS
from opharm.stats.lock import analysis_rows

MODELS = ("qwen35_4b", "qwen35_9b", "qwen3_4b_2507", "gemma4_e4b", "llama31_8b")
FROZEN = ("src/opharm/bench/evaluate.py", "src/opharm/bench/labels.py")


def frozen_commit():
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *FROZEN], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"commit the evaluator before drawing: {dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def ticket(m):
    return "".join(t for _, t in m["user"])


def item(model, source, m, text, max_new, condition=""):
    return {"model": model, "source": source, "instance": m["id"], "condition": condition, "ticket": ticket(m), "output": text,
            "label_v21": label(text, m, model, max_new)["label"], "label_strict": label(text, m, model, max_new, strict=True)["label"],
            "capped": bool(max_new) and capped(text, model, max_new)}


def grid_pool(model, meta, seen):
    rows = [{**meta[r["id"]], **r} for r in map(json.loads, open(RUNS / model / "grid" / "results.jsonl"))
            if r["id"] in meta and meta[r["id"]]["set"] == "main" and "text" in r]
    rows = [r for r in analysis_rows(rows, model) if (model, r["id"]) not in seen]
    dpc = [r for r in rows if r["target"] + r["env"] + r["policy"] == "DPC"]
    other = [r for r in rows if r["target"] + r["env"] + r["policy"] != "DPC"]
    return dpc, other


def secondary_pool(meta, seen):
    pool = []
    for model, tag in (("qwen35_4b", "think"), ("qwen35_4b", "think_rest"), ("qwen35_4b", "narr"), ("qwen35_4b", "rules"),
                       ("qwen35_4b", "rules_min"), ("qwen3_4b_2507", "rules"), ("qwen3_4b_2507", "rules_min")):
        path = RUNS / model / tag / "results.jsonl"
        if not path.exists():
            continue
        rows = [{**meta[r["id"]], **r} for r in map(json.loads, open(path)) if r["id"] in meta and "text" in r]
        pool += [("run:" + tag, model, r, r["text"], run_max_new(model, tag), "") for r in analysis_rows(rows, model) if (model, r["id"]) not in seen]
    for mode in ("c2", "steer", "gain"):
        path = RUNS / "qwen35_4b" / "grid" / f"decoded_{mode}.jsonl"
        info = json.loads((RUNS / "qwen35_4b" / "grid" / f"decoded_{mode}_meta.json").read_text())["args"]["max_new"]
        for r in map(json.loads, open(path)):
            if r["condition"] not in ("clean", "baseline", "identity"):
                pool.append(("decoded:" + mode, "qwen35_4b", meta[r["id"]], r["text"], info, r["condition"]))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit3")
    ap.add_argument("--prefix", default="C")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--dpc", type=int, default=12)
    ap.add_argument("--other", type=int, default=8)
    ap.add_argument("--secondary", type=int, default=20)
    args = ap.parse_args()
    commit = frozen_commit()
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
    for f in ("variants.jsonl", "variants_min.jsonl", "variants_chk.jsonl"):
        if (BENCH / f).exists():
            meta.update({m["id"]: m for m in map(json.loads, open(BENCH / f))})
    seen = set()
    for key in ("audit/audit_key.csv", "audit2/audit_key.csv"):
        seen |= {(r["model"], r["instance"]) for r in csv.DictReader(open(RUNS / key, newline=""))}
    rng, items, sizes = random.Random(args.seed), [], {}
    for model in MODELS:
        dpc, other = grid_pool(model, meta, seen)
        max_new = run_max_new(model, "grid")
        for name, pool, n in (("dpc", dpc, args.dpc), ("other_main", other, args.other)):
            sizes[f"{model}:{name}"] = len(pool)
            items += [dict(item(model, "grid", r, r["text"], max_new), stratum=name, population=len(pool)) for r in rng.sample(pool, n)]
    sec = secondary_pool(meta, seen)
    group = lambda source: "think" if "think" in source else "narr" if "narr" in source else "decoded" if source.startswith("decoded") else "rules"
    groups = {g: [x for x in sec if group(x[0]) == g] for g in ("think", "narr", "decoded", "rules")}
    sizes.update({f"secondary:{g}": len(v) for g, v in groups.items()})
    for g, pool in groups.items():
        for source, model, r, text, max_new, cond in rng.sample(pool, min(len(pool), args.secondary // len(groups))):
            items.append(dict(item(model, source, r, text, max_new, cond), stratum="secondary:" + g, population=len(pool)))
    rng.shuffle(items)
    out = RUNS / args.out
    out.mkdir(exist_ok=True)
    with open(out / "audit_items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "ticket", "model_output", "label_annotator", "notes"])
        for i, x in enumerate(items):
            w.writerow([f"{args.prefix}{i:03d}", x["ticket"], x["output"], "", ""])
    cols = ["model", "source", "instance", "condition", "stratum", "population", "label_v21", "label_strict", "capped"]
    with open(out / "audit_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item"] + cols)
        for i, x in enumerate(items):
            w.writerow([f"{args.prefix}{i:03d}"] + [x[c] for c in cols])
    manifest = {"evaluator": VERSION, "commit": commit, "seed": args.seed, "n": len(items), "strata_sizes": sizes,
                "design": {"dpc_per_model": args.dpc, "other_main_per_model": args.other, "secondary": args.secondary},
                "excluded_batches": ["audit", "audit2"], "by_stratum": dict(Counter(x["stratum"] for x in items))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("evaluator", "commit", "n", "by_stratum")}))


if __name__ == "__main__":
    main()
