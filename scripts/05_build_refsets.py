import argparse
import json
from collections import Counter

import opharm
import numpy as np

from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.paths import BENCH
from opharm.refsets.content import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_4b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tok = load_tokenizer(args.model)
    items = build(tok, args.seed)
    for it in items:
        r = render(tok, [(None, it["system"])], [(None, it["user"])], TOOLS)
        it["n_tokens"], it["t_inst"], it["t_post"] = len(r.ids), r.t_inst, r.t_post
        assert tok.decode(r.ids[r.t_inst]) == "."
    out = BENCH / "refsets" / "content.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(Counter((it["split"], it["kind"]) for it in items))
    for kind in ("harmful", "benign"):
        n = [it["n_tokens"] for it in items if it["kind"] == kind]
        print(kind, "prompt tokens median", int(np.median(n)), "range", min(n), max(n))


if __name__ == "__main__":
    main()
