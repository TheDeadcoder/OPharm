import argparse
import hashlib
import importlib
import json
import sys
import time

import opharm

from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.paths import BENCH, ROOT, RUNS

sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("09_run_behavior")


def report(args):
    from collections import Counter
    from opharm.bench.labels import label, tokenizer
    from opharm.paths import RESULTS
    from opharm.stats.lock import analysis_rows
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / args.instances))}
    src = RUNS / args.model / args.tag
    budget = json.loads((src / "manifest.json").read_text())["args"]["max_new"]
    cont = {r["id"]: r for r in map(json.loads, open(RUNS / args.model / f"{args.tag}_cont" / "results.jsonl"))}
    rows = analysis_rows([{**meta[r["id"]], **r} for r in map(json.loads, open(src / "results.jsonl"))
                          if r["id"] in meta and meta[r["id"]]["set"] == "main" and "text" in r], args.model)
    cell = lambda r: r["target"] + r["env"] + ("|" + r["variant"] if "variant" in r else r["policy"])
    trans, before, after = Counter(), {}, {}
    for r in rows:
        b = a = label(r["text"], r, args.model, budget)["label"]
        if r["id"] in cont:
            c = cont[r["id"]]
            a = label(c["text"], r, args.model, budget + args.max_more if c["capped_again"] else None)["label"]
            trans[(r["split"], cell(r), b, a)] += 1
        before[r["id"]], after[r["id"]] = b, a
    out = {"model": args.model, "budget": budget, "max_more": args.max_more, "continued": sum(r["id"] in cont for r in rows),
           "still_capped": sum(c["capped_again"] for c in cont.values()), "rates": {},
           "transitions": [{"split": s, "cell": c, "before": b, "after": a, "n": n} for (s, c, b, a), n in sorted(trans.items())]}
    tok = tokenizer(args.model)
    curves = {}
    for extra in [int(b) for b in args.budgets.split(",") if b]:
        at = {}
        for r in rows:
            if r["id"] not in cont:
                at[r["id"]] = before[r["id"]]
                continue
            ids = tok.encode(cont[r["id"]]["text"], add_special_tokens=False)
            at[r["id"]] = after[r["id"]] if len(ids) < extra else label(tok.decode(ids[:extra]), r, args.model, extra)["label"]
        curves[f"budget_{extra}"] = at
    states = (("before", before), ("after", after)) + tuple(curves.items())
    for split in ("dev", "heldout"):
        for c in sorted({cell(r) for r in rows}):
            sub = [r for r in rows if r["split"] == split and cell(r) == c]
            if sub:
                out["rates"][f"{split}_{c}"] = {"n": len(sub), **{f"{k}_{lab}": round(sum(d[r["id"]] == lab for r in sub) / len(sub), 4)
                                                                 for k, d in states for lab in ("EXEC_MATCH", "ASK", "INCOMPLETE", "DECLINE")}}
    RESULTS.mkdir(exist_ok=True)
    name = f"continuation_{args.model}" + ("" if args.tag == "grid" else f"_{args.tag}")
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["rates"].items() if k.split("_", 1)[1].startswith("DP")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--max-more", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--instances", default="instances.jsonl")
    ap.add_argument("--budgets", default="512")
    args = ap.parse_args()
    if args.report:
        return report(args)
    src = RUNS / args.model / args.tag
    src_args = json.loads((src / "manifest.json").read_text())["args"]
    budget = src_args["max_new"]
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / args.instances))}
    tok = load_tokenizer(args.model)
    todo = []
    for r in map(json.loads, open(src / "results.jsonl")):
        if "text" not in r or meta[r["id"]]["set"] not in runner.DECODE_SETS:
            continue
        reply = tok.encode(r["text"], add_special_tokens=False)
        if len(reply) >= budget - 1:
            if tok.decode(reply) != r["text"]:
                raise ValueError(f"{r['id']}: reply does not round-trip through the tokenizer")
            m = meta[r["id"]]
            prompt = render(tok, m["system"], m["user"], runner.tools_for(m, bool(src_args.get("no_tools"))), thinking=bool(src_args.get("thinking")),
                            tools_in_user=not src_args.get("tools_in_system")).ids
            todo.append((r, prompt + reply, len(reply)))
    todo = todo[:args.limit] if args.limit else todo
    print(json.dumps({"model": args.model, "budget": budget, "capped": len(todo)}))
    if args.dry_run:
        return
    out = RUNS / args.model / f"{args.tag}_cont"
    out.mkdir(parents=True, exist_ok=True)
    from opharm.models import load_model
    from opharm.run.generate import CLOSERS, greedy
    model = load_model(args.model)
    t0 = time.time()
    texts = greedy(model, tok, [ids for _, ids, _ in todo], args.max_more, args.batch, extra_stops=CLOSERS)
    with open(out / "results.jsonl", "w") as f:
        for (r, ids, n), more in zip(todo, texts):
            again = len(tok.encode(more, add_special_tokens=False)) >= args.max_more - 1
            f.write(json.dumps({"id": r["id"], "reply_tokens": n, "continuation": more, "text": r["text"] + more, "capped_again": again}) + "\n")
    manifest = {"model": args.model, "source": args.tag, "budget": budget, "args": vars(args), "git": runner.git_state(), "n": len(todo),
                "source_sha256": hashlib.sha256(open(src / "results.jsonl", "rb").read()).hexdigest(), "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n", "seconds")}))


if __name__ == "__main__":
    main()
