import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict

import opharm
import numpy as np
import torch

from opharm.bench.oracle import label
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, pins, render
from opharm.models import load_model
from opharm.paths import BENCH, ROOT, RUNS
from opharm.run.cache import forward_capture
from opharm.run.generate import greedy

DECODE_SETS = {"main", "ladder", "cue", "shortcut"}


def yes_no_ids(tok):
    out = {}
    for key, words in (("yes", ("Yes", "yes", "YES")), ("no", ("No", "no", "NO"))):
        out[key] = [ids[0] for w in words if len(ids := tok.encode(w, add_special_tokens=False)) == 1]
    return out


def stratified(rows, n):
    by_cls = defaultdict(list)
    for cls, sk in sorted({(r["cls"], r["skeleton"]) for r in rows}, key=lambda t: (t[0], int(t[1].rsplit(".", 1)[1]), t[1])):
        by_cls[cls].append(sk)
    picked, i = [], 0
    while len(picked) < n and any(i < len(v) for v in by_cls.values()):
        picked += [v[i] for v in by_cls.values() if i < len(v)][: n - len(picked)]
        i += 1
    return set(picked)


def git_state():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "uncommitted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="main,judge,shortcut")
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--questions", default="q1,q2")
    ap.add_argument("--skeletons", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-acts", action="store_true")
    args = ap.parse_args()
    sets, questions = set(args.sets.split(",")), set(args.questions.split(","))

    rows = [json.loads(line) for line in open(BENCH / "instances.jsonl")]
    rows = [r for r in rows if r["set"] in sets and (args.split == "all" or r["split"] == args.split)
            and (r["set"] != "judge" or r["question"] in questions)]
    if args.skeletons:
        keep = stratified(rows, args.skeletons)
        rows = [r for r in rows if r["skeleton"] in keep]
    out = RUNS / args.model / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tok, model = load_model_and_tok(args.model)
    opener, yn = tok.convert_tokens_to_ids("<tool_call>"), yes_no_ids(tok)
    t0 = time.time()
    ids, results, acts = [], [], None
    for k, r in enumerate(rows):
        rr = render(tok, r["system"], r["user"], None if r["set"] == "flat" else TOOLS)
        m, logits, a = forward_capture(model, rr.ids, [rr.t_inst, rr.t_post], opener)
        if acts is None and not args.no_acts:
            acts = np.lib.format.open_memmap(out / "acts.npy", mode="w+", dtype=np.float32, shape=(len(rows), *a.shape))
        if acts is not None:
            acts[k] = a.numpy()
        lp = logits.log_softmax(-1)
        top = lp.topk(5)
        res = {"id": r["id"], "set": r["set"], "n_tokens": len(rr.ids), "m": m,
               "top5": [[tok.decode([i]), round(v, 3)] for v, i in zip(top.values.tolist(), top.indices.tolist())]}
        if r["set"] == "judge":
            py, pn = torch.logsumexp(lp[yn["yes"]], 0), torch.logsumexp(lp[yn["no"]], 0)
            res["judge"], res["yn_mass"] = (py - pn).item(), (py.exp() + pn.exp()).item()
        results.append(res)
        ids.append(rr.ids)
    prefill_s = time.time() - t0
    if acts is not None:
        acts.flush()
    dec = [k for k, r in enumerate(rows) if r["set"] in DECODE_SETS]
    t1 = time.time()
    texts = greedy(model, tok, [ids[k] for k in dec], args.max_new, args.batch, extra_stops=("</tool_call>",))
    for k, text in zip(dec, texts):
        results[k]["text"], results[k]["label"] = text, label(text, rows[k]["oracle"])
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    manifest = {
        "model": args.model, "revision": pins()[args.model]["revision"], "args": vars(args), "git": git_state(),
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        "benchmark_sha256": json.loads((BENCH / "manifest.json").read_text())["sha256_instances"],
        "n": len(rows), "n_decoded": len(dec), "acts_shape": None if acts is None else list(acts.shape),
        "seconds": {"prefill": round(prefill_s, 1), "decode": round(time.time() - t1, 1)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["seconds"]), "n", len(rows), "decoded", len(dec))


def load_model_and_tok(key):
    return load_tokenizer(key), load_model(key)


if __name__ == "__main__":
    main()
