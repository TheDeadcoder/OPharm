import argparse
import gc
import json
import time

import opharm
import numpy as np
import torch

from opharm.bench.smoke import smoke_prompts
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, pins, render, snapshot_dir
from opharm.models import load_model
from opharm.paths import RESULTS
from opharm.run.decision import action_logodds

GEN = 32


def free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def mps_generate(key, prompts, stop, solver):
    model = load_model(key, solver=solver)
    conts, secs = [], []
    with torch.inference_mode():
        for ids in prompts:
            x = torch.tensor([ids], device=model.device)
            g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=GEN, do_sample=False,
                               eos_token_id=stop, pad_token_id=stop)
            conts.append(g[0, len(ids):].tolist())
            t = time.perf_counter()
            model(input_ids=x, use_cache=False, logits_to_keep=1).logits.float().cpu()
            secs.append(time.perf_counter() - t)
    del model
    free()
    return conts, secs


def torch_scores(key, device, dtype, prompts, conts, solver=None):
    model = load_model(key, device=device, dtype=dtype, solver=solver)
    out, secs = [], []
    with torch.inference_mode():
        for ids, cont in zip(prompts, conts):
            x = torch.tensor([ids + cont], device=model.device)
            t = time.perf_counter()
            out.append(model(input_ids=x, use_cache=False, logits_to_keep=len(cont) + 1).logits[0].float().cpu())
            secs.append(time.perf_counter() - t)
    del model
    free()
    return out, secs


def mlx_scores(key, prompts, conts):
    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(str(snapshot_dir(key)))
    out, secs = [], []
    for ids, cont in zip(prompts, conts):
        t = time.perf_counter()
        z = model(mx.array([ids + cont]))[0, len(ids) - 1:].astype(mx.float32)
        mx.eval(z)
        secs.append(time.perf_counter() - t)
        out.append(torch.from_numpy(np.array(z)))
    del model
    gc.collect()
    mx.clear_cache()
    return out, secs


def first_divergence(rows, cont):
    pred = rows[:len(cont)].argmax(-1).tolist()
    return next((i for i, (a, b) in enumerate(zip(pred, cont)) if a != b), None)


def compare(ref, test, conts, opener):
    lr = torch.stack([r[0] for r in ref])
    lt = torch.stack([r[0] for r in test])
    diff = (lr - lt).abs().max(-1).values
    pr, pt = lr.log_softmax(-1), lt.log_softmax(-1)
    div = [first_divergence(r, c) for r, c in zip(test, conts)]
    return {
        "n": len(diff),
        "top1_agree": int((lr.argmax(-1) == lt.argmax(-1)).sum()),
        "max_abs_logit_diff": {"max": diff.max().item(), "median": diff.median().item()},
        "m_abs_diff_max": (action_logodds(lr, opener) - action_logodds(lt, opener)).abs().max().item(),
        "kl_max": (pr.exp() * (pr - pt)).sum(-1).max().item(),
        "greedy_identical_to_mps": sum(d is None for d in div),
        "greedy_first_divergence": div,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ref-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--skip-mlx", action="store_true")
    ap.add_argument("--solver", default="loop", choices=["default", "loop"])
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    opener, stop = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("<|im_end|>")
    prompts = [render(tok, s, u, TOOLS).ids for s, u in smoke_prompts()[:args.n]]
    conts, mps_secs = mps_generate(args.model, prompts, stop, args.solver)
    mps_scores = torch_scores(args.model, "mps", torch.bfloat16, prompts, conts, args.solver)
    ref_name = f"cpu_{args.ref_dtype}"
    scores = {ref_name: torch_scores(args.model, "cpu", getattr(torch, args.ref_dtype), prompts, conts),
              "mps_bfloat16": mps_scores}
    if not args.skip_mlx:
        scores["mlx_bfloat16"] = mlx_scores(args.model, prompts, conts)

    comparisons = {f"{k}_vs_{ref_name}": compare(scores[ref_name][0], v[0], conts, opener)
                   for k, v in scores.items() if k != ref_name}
    comparisons[f"{ref_name}_greedy_vs_mps"] = {
        "greedy_identical_to_mps": sum(first_divergence(r, c) is None for r, c in zip(scores[ref_name][0], conts)),
        "greedy_first_divergence": [first_divergence(r, c) for r, c in zip(scores[ref_name][0], conts)],
    }
    report = {
        "model": args.model, "revision": pins()[args.model]["revision"], "solver": args.solver, "n": len(prompts),
        "prompt_tokens": {"min": min(map(len, prompts)), "max": max(map(len, prompts))},
        "median_seconds_prefill_mps": float(np.median(mps_secs[1:])),
        "median_seconds_scoring_pass": {k: float(np.median(v[1][1:])) for k, v in scores.items()},
        "m_reference": action_logodds(torch.stack([r[0] for r in scores[ref_name][0]]), opener).tolist(),
        "comparisons": comparisons,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"numerics_{args.model}_{args.solver}.json").write_text(json.dumps(report, indent=1))
    for k, v in comparisons.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "greedy_first_divergence"})
    print("prompt_tokens", report["prompt_tokens"], "prefill_mps", round(report["median_seconds_prefill_mps"], 3),
          "scoring", {k: round(v, 3) for k, v in report["median_seconds_scoring_pass"].items()})


if __name__ == "__main__":
    main()
