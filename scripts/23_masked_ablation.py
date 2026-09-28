import argparse
import json

import opharm
import numpy as np
import torch

from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, render
from opharm.interp import hooks
from opharm.interp.directions import outlier_keep
from opharm.models import load_model
from opharm.paths import RESULTS, RUNS
from opharm.refsets.refusal import is_refusal
from opharm.run.generate import greedy


def rate(model, tok, ids, edits, max_new):
    with hooks.hooked(*edits):
        return float(np.mean([is_refusal(t) for t in greedy(model, tok, ids, max_new)]))


def last_logprobs(model, ids, edits=()):
    out = []
    with torch.inference_mode(), hooks.hooked(*edits):
        for x in ids:
            z = model(input_ids=torch.tensor([x], device=model.device), use_cache=False, logits_to_keep=1).logits[0, -1]
            out.append(z.float().log_softmax(-1).cpu())
    return torch.stack(out)


def random_like(v, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    u = v / v.norm()
    r = r - (r @ u) * u
    return r / r.norm() * v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ratio", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--eval-new", type=int, default=64)
    args = ap.parse_args()
    run = RUNS / args.model / "refsets"
    tok, model = load_tokenizer(args.model), load_model(args.model)
    items = [json.loads(line) for line in open(run / "items.jsonl")]
    dirs = np.load(run / "directions.npz")
    ids = {it["id"]: render(tok, [(None, it["system"])], [(None, it["user"])], TOOLS).ids for it in items}
    pick = lambda split, kind: [ids[it["id"]] for it in items if it["split"] == split and it["kind"] == kind]
    sel_b, ev_h = pick("select", "benign"), pick("eval", "harmful")
    keep = [torch.tensor(k) for k in outlier_keep(args.model, args.ratio)]
    layer = json.loads((RESULTS / f"c5_{args.model}.json").read_text())["selected_layer"]
    r = torch.tensor(dirs["r_ref"][layer])
    rand = [random_like(r, s) for s in range(args.seeds)]
    base_b = last_logprobs(model, sel_b)
    kl = lambda edits: float((base_b.exp() * (base_b - last_logprobs(model, sel_b, edits))).sum(-1).mean())
    masked = lambda v: [hooks.ablate_masked(model, v, keep)]
    n_layers = len(hooks.layers(model))
    res = {"model": args.model, "ratio": args.ratio, "layer": layer, "excluded_per_point": [int((~k).sum()) for k in keep],
           "baseline_refusal_harmful_eval": rate(model, tok, ev_h, [], args.eval_new),
           "direction": {"refusal": rate(model, tok, ev_h, masked(r), args.eval_new), "kl_benign": kl(masked(r))},
           "random_masked": [{"refusal": rate(model, tok, ev_h, masked(q), args.eval_new), "kl_benign": kl(masked(q))} for q in rand],
           "random_unmasked_kl_benign": [kl([hooks.ablate(model, q)]) for q in rand[:2]],
           "kl_scan_r_ref_masked": [{"layer": l, "kl": kl(masked(torch.tensor(dirs["r_ref"][l])))}
                                    for l in range(int(0.2 * n_layers), int(0.8 * n_layers))]}
    base = res["baseline_refusal_harmful_eval"]
    rr, rk = [x["refusal"] for x in res["random_masked"]], [x["kl_benign"] for x in res["random_masked"]]
    res["criteria"] = {"random_refusal_within_10_points": bool(abs(np.mean(rr) - base) <= 0.10), "random_kl_below_0_1": bool(np.mean(rk) < 0.1)}
    res["valid"] = all(res["criteria"].values())
    res["effect_vs_random_pts"] = 100 * (res["direction"]["refusal"] - float(np.mean(rr)))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"c5_masked_{args.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("baseline_refusal_harmful_eval", "direction", "criteria", "valid", "effect_vs_random_pts")}, indent=1))


if __name__ == "__main__":
    main()
