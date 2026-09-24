import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import opharm
from opharm.bench import generate
from opharm.bench.align import pair_exclusion
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.paths import BENCH

MODELS = ("qwen35_4b", "qwen3_4b_2507")
PAIRS = {"env": ("env", "PS", 1), "target": ("cmd", "DB", 0), "rollback": ("rollback", "AN", 2), "policy": ("policy", "CN", 3)}


def rendered(tok, row):
    return render(tok, row["system"], row["user"], None if row["set"] == "flat" else TOOLS)


def check_pairs(tok, rows):
    main = {r["id"]: rendered(tok, r) for r in rows if r["set"] == "main"}
    by_code = {(r["skeleton"], r["target"] + r["env"] + r["rollback"] + r["policy"]): r["id"] for r in rows if r["set"] == "main"}
    results, exclusions = Counter(), []
    for (skeleton, code), rid in by_code.items():
        for name, (span, levels, pos) in PAIRS.items():
            if code[pos] != levels[0]:
                continue
            other = by_code[(skeleton, code[:pos] + levels[1] + code[pos + 1:])]
            reason = pair_exclusion(main[rid], main[other], span)
            results[(name, reason is None)] += 1
            if reason:
                exclusions.append({"skeleton": skeleton, "pair": name, "a": rid, "b": other, "reason": reason})
    return results, exclusions


def main():
    toks = {k: load_tokenizer(k) for k in MODELS}
    skeletons = generate.plan_skeletons(list(toks.values()))
    rows = generate.expand(skeletons, generate.tools_prose(TOOLS))
    manifest = {"n_skeletons": len(skeletons), "n_instances": len(rows),
                "by_set_split": {f"{s}/{p}": n for (s, p), n in sorted(Counter((r["set"], r["split"]) for r in rows).items())},
                "skeleton_forms": dict(Counter((s["split"], s["form"]) for s in skeletons)), "alignment": {}}
    manifest["skeleton_forms"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(manifest["skeleton_forms"].items())}
    all_exclusions = []
    for key, tok in toks.items():
        bad = [r["id"] for r in rows if tok.decode([rendered(tok, r).ids[rendered(tok, r).t_inst]]) != "."]
        if bad:
            raise ValueError(f"{key}: t_inst is not the closing period for {bad[:3]}")
        results, exclusions = check_pairs(tok, rows)
        manifest["alignment"][key] = {name: {"aligned": results[(name, True)], "excluded": results[(name, False)]}
                                      for name in PAIRS}
        all_exclusions += [dict(e, model=key) for e in exclusions]
    BENCH.mkdir(exist_ok=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    (BENCH / "instances.jsonl").write_text(body)
    (BENCH / "skeletons.jsonl").write_text("".join(json.dumps(s) + "\n" for s in skeletons))
    manifest["sha256_instances"] = hashlib.sha256(body.encode()).hexdigest()
    (BENCH / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with open(BENCH / "exclusions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "skeleton", "pair", "a", "b", "reason"])
        w.writeheader()
        w.writerows(all_exclusions)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256_instances"}, indent=1))
    print("exclusion reasons:", Counter((e["model"], e["pair"], e["reason"]) for e in all_exclusions).most_common(12))


if __name__ == "__main__":
    main()
