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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--max-more", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    src = RUNS / args.model / args.tag
    budget = json.loads((src / "manifest.json").read_text())["args"]["max_new"]
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
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
            prompt = render(tok, m["system"], m["user"], runner.tools_for(m, False)).ids
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
