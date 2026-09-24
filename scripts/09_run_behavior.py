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

DECODE_SETS = {"main", "narr", "ladder", "cue", "shortcut"}


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


def done_ids(path):
    return {json.loads(line)["id"] for line in open(path)} if path.exists() else set()


def ids_of(tok, row, thinking):
    return render(tok, row["system"], row["user"], None if row["set"] == "flat" else TOOLS, thinking=thinking).ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="main,judge,shortcut")
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--questions", default="q1,q2,q3")
    ap.add_argument("--skeletons", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--no-acts", action="store_true")
    ap.add_argument("--thinking", action="store_true")
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
    order = [r["id"] for r in rows]
    if (out / "order.json").exists() and json.loads((out / "order.json").read_text()) != order:
        raise ValueError("run directory holds a different instance order; use a new tag")
    (out / "order.json").write_text(json.dumps(order))

    tok, model = load_tokenizer(args.model), load_model(args.model)
    opener, yn = tok.convert_tokens_to_ids("<tool_call>"), yes_no_ids(tok)
    acts = None
    if not args.no_acts:
        shape = (len(rows), 2, len(model.model.layers) + 1, model.config.hidden_size)
        mode = "r+" if (out / "acts.npy").exists() else "w+"
        acts = np.lib.format.open_memmap(out / "acts.npy", mode=mode, dtype=np.float32, shape=shape)

    t0, done = time.time(), done_ids(out / "prefill.jsonl")
    with open(out / "prefill.jsonl", "a") as f:
        for k, r in enumerate(rows):
            if r["id"] in done:
                continue
            rr = render(tok, r["system"], r["user"], None if r["set"] == "flat" else TOOLS, thinking=args.thinking)
            m, logits, a = forward_capture(model, rr.ids, [rr.t_inst, rr.t_post], opener)
            if acts is not None:
                acts[k] = a.numpy()
            lp = logits.log_softmax(-1)
            top = lp.topk(5)
            res = {"id": r["id"], "row": k, "n_tokens": len(rr.ids), "m": m,
                   "top5": [[tok.decode([i]), round(v, 3)] for v, i in zip(top.values.tolist(), top.indices.tolist())]}
            if r["set"] == "judge":
                py, pn = torch.logsumexp(lp[yn["yes"]], 0), torch.logsumexp(lp[yn["no"]], 0)
                res["judge"], res["yn_mass"] = (py - pn).item(), (py.exp() + pn.exp()).item()
            f.write(json.dumps(res) + "\n")
            if k % args.chunk == 0:
                f.flush()
                if acts is not None:
                    acts.flush()
    if acts is not None:
        acts.flush()
    prefill_s = time.time() - t0

    t1 = time.time()
    for name, pick, max_new, batch, stops in (
        ("decode", lambda r: r["set"] in DECODE_SETS, args.max_new, args.batch, ("</tool_call>",)),
        ("judge", lambda r: r["set"] == "judge", 4, args.batch * 2, ()),
    ):
        done = done_ids(out / f"{name}.jsonl")
        todo = [r for r in rows if pick(r)]
        with open(out / f"{name}.jsonl", "a") as f:
            for s in range(0, len(todo), args.chunk):
                part = todo[s:s + args.chunk]
                if all(r["id"] in done for r in part):
                    continue
                texts = greedy(model, tok, [ids_of(tok, r, args.thinking) for r in part], max_new, batch, extra_stops=stops)
                for r, text in zip(part, texts):
                    if r["id"] in done:
                        continue
                    if name == "decode":
                        rec = {"id": r["id"], "text": text, "label": label(text, r["oracle"])}
                    else:
                        word = text.strip().split()[0].strip(".,:!*").lower() if text.strip() else ""
                        rec = {"id": r["id"], "answer": word if word in ("yes", "no") else "other"}
                    f.write(json.dumps(rec) + "\n")
                f.flush()

    merged = {json.loads(line)["id"]: json.loads(line) for line in open(out / "prefill.jsonl")}
    for name in ("decode", "judge"):
        if (out / f"{name}.jsonl").exists():
            for line in open(out / f"{name}.jsonl"):
                rec = json.loads(line)
                merged[rec["id"]].update(rec)
    (out / "results.jsonl").write_text("".join(json.dumps(dict(merged[i], set=r["set"])) + "\n" for i, r in zip(order, rows)))
    manifest = {
        "model": args.model, "revision": pins()[args.model]["revision"], "args": vars(args), "git": git_state(),
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        "benchmark_sha256": json.loads((BENCH / "manifest.json").read_text())["sha256_instances"],
        "n": len(rows), "acts_shape": None if acts is None else list(acts.shape),
        "seconds_this_session": {"prefill": round(prefill_s, 1), "decode": round(time.time() - t1, 1)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["seconds_this_session"]), "n", len(rows))


if __name__ == "__main__":
    main()
