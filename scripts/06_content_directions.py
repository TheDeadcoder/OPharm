import argparse
import json
from collections import Counter

import opharm
import numpy as np
from sklearn.metrics import roc_auc_score

from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, pins, render
from opharm.models import load_model
from opharm.paths import BENCH, RESULTS, RUNS
from opharm.refsets.refusal import is_refusal
from opharm.run.cache import forward_capture
from opharm.run.generate import greedy


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    out = RUNS / args.model / "refsets"
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model)
    model = load_model(args.model)
    opener = tok.convert_tokens_to_ids("<tool_call>")
    items = [json.loads(line) for line in open(BENCH / "refsets" / "content.jsonl")]
    rendered = [render(tok, [(None, it["system"])], [(None, it["user"])], TOOLS) for it in items]

    acts = []
    for it, r in zip(items, rendered):
        it["m"], _, a = forward_capture(model, r.ids, [r.t_inst, r.t_post], opener)
        acts.append(a.numpy())
    acts = np.stack(acts)
    for it, text in zip(items, greedy(model, tok, [r.ids for r in rendered], args.max_new)):
        it["text"], it["refused"] = text, is_refusal(text)
    np.save(out / "acts.npy", acts)
    (out / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))

    kind, split = np.array([it["kind"] for it in items]), np.array([it["split"] for it in items])
    refused = np.array([it["refused"] for it in items])
    ext = split == "extract"
    harm, ok = ext & (kind == "harmful") & refused, ext & (kind == "benign") & ~refused
    dirs = {
        "r_harm": acts[harm, 0].mean(0) - acts[ok, 0].mean(0),
        "r_ref": acts[harm, 1].mean(0) - acts[ok, 1].mean(0),
        "r_harm_unfiltered": acts[ext & (kind == "harmful"), 0].mean(0) - acts[ext & (kind == "benign"), 0].mean(0),
        "r_ref_unfiltered": acts[ext & (kind == "harmful"), 1].mean(0) - acts[ext & (kind == "benign"), 1].mean(0),
        "mu_harm_inst": acts[harm, 0].mean(0), "mu_benign_inst": acts[ok, 0].mean(0),
    }
    np.savez(out / "directions.npz", **dirs)

    ev = split == "eval"
    y = (kind[ev] == "harmful").astype(int)
    auroc = {name: [float(roc_auc_score(y, acts[ev, pos, p] @ unit(dirs[name][p]))) for p in range(acts.shape[2])]
             for name, pos in (("r_harm", 0), ("r_ref", 1))}
    cos = (unit(dirs["r_harm"]) * unit(dirs["r_ref"])).sum(-1)
    h = acts[ev, 0]
    belief = ((h * unit(dirs["mu_harm_inst"])[None]).sum(-1) / np.linalg.norm(h, axis=-1)
              - (h * unit(dirs["mu_benign_inst"])[None]).sum(-1) / np.linalg.norm(h, axis=-1)).mean(-1)
    first = {k: Counter(tok.convert_ids_to_tokens(tok(it["text"], add_special_tokens=False)["input_ids"][:1])[0]
                        if it["text"] else "<empty>" for it in items if it["refused"] == v).most_common(8)
             for k, v in (("refused", True), ("not_refused", False))}
    summary = {
        "model": args.model, "revision": pins()[args.model]["revision"],
        "n_direction_examples": {"harmful_refused": int(harm.sum()), "benign_accepted": int(ok.sum())},
        "refusal_rate": {f"{k}_{s}": float(refused[(kind == k) & (split == s)].mean())
                         for k in ("harmful", "benign") for s in ("extract", "select", "eval")},
        "m_median": {k: float(np.median([it["m"] for it in items if it["kind"] == k])) for k in ("harmful", "benign")},
        "first_tokens": first,
        "auroc_eval": auroc,
        "cos_harm_ref": cos.tolist(),
        "zhao_belief_eval": {"harmful": float(belief[y == 1].mean()), "benign": float(belief[y == 0].mean())},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"refsets_{args.model}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("auroc_eval", "cos_harm_ref")}, indent=1))
    best = {k: (int(np.argmax(v)), round(max(v), 3)) for k, v in auroc.items()}
    print("best eval AUROC (point, value):", best, "| cos(r_harm, r_ref) at those points:",
          [round(float(cos[p]), 3) for p, _ in best.values()])


if __name__ == "__main__":
    main()
