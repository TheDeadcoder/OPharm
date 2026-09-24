import argparse
import gc
import json
import time
from contextlib import contextmanager

import opharm
import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qm

from opharm.bench.smoke import smoke_prompts
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, pins, render, snapshot_dir
from opharm.models import load_model
from opharm.paths import RESULTS

DECODE = 64


@contextmanager
def gdn_solver(mode):
    solve, exporting = torch.linalg.solve_triangular, qm.is_torchdynamo_exporting
    if mode == "cpu_solve":
        torch.linalg.solve_triangular = lambda a, b, **kw: solve(a.cpu(), b.cpu(), **kw).to(a.device)
    elif mode == "loop":
        qm.is_torchdynamo_exporting = lambda: True
    try:
        yield
    finally:
        torch.linalg.solve_triangular, qm.is_torchdynamo_exporting = solve, exporting


def timed(fn, passes):
    fn()
    fn()
    secs = []
    for _ in range(passes):
        t = time.perf_counter()
        fn()
        secs.append(time.perf_counter() - t)
    return float(np.median(secs)), float(np.percentile(secs, 90))


def torch_modes(key, prompts, passes, stop):
    model = load_model(key, solver="default")
    x = torch.tensor([prompts[0]], device=model.device)
    xb = torch.tensor([prompts[0]] * 4, device=model.device)
    out, ref = {}, None
    with torch.inference_mode():
        for mode in ("default", "cpu_solve", "loop"):
            with gdn_solver(mode):
                logits = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
                ref = logits if ref is None else ref
                prefill = timed(lambda: model(input_ids=x, use_cache=False, logits_to_keep=1).logits.float().cpu(), passes)
                batch4 = timed(lambda: model(input_ids=xb, use_cache=False, logits_to_keep=1).logits.float().cpu(), max(3, passes // 5))
                t = time.perf_counter()
                g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=DECODE, min_new_tokens=DECODE,
                                   do_sample=False, pad_token_id=stop)
                decode_s = time.perf_counter() - t - prefill[0]
            out[f"mps_{mode}"] = {
                "prefill_s_median_p90": prefill,
                "batch4_s_per_prompt": batch4[0] / 4,
                "decode_tok_per_s": (g.shape[1] - x.shape[1]) / decode_s,
                "max_abs_logit_diff_vs_default": (logits - ref).abs().max().item(),
            }
        out["mps_peak_driver_gb"] = torch.mps.driver_allocated_memory() / 1e9
    del model
    gc.collect()
    torch.mps.empty_cache()
    return out


def mlx_mode(key, prompts, passes):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, _ = load(str(snapshot_dir(key)))
    x = mx.array([prompts[0]])
    xb = mx.array([prompts[0]] * 4)

    def fwd(arr):
        z = model(arr)[:, -1]
        mx.eval(z)

    prefill = timed(lambda: fwd(x), passes)
    batch4 = timed(lambda: fwd(xb), max(3, passes // 5))
    t = time.perf_counter()
    n = sum(1 for _ in generate_step(mx.array(prompts[0]), model, max_tokens=DECODE, sampler=lambda z: mx.argmax(z, axis=-1)))
    decode_s = time.perf_counter() - t - prefill[0]
    out = {"prefill_s_median_p90": prefill, "batch4_s_per_prompt": batch4[0] / 4, "decode_tok_per_s": n / decode_s,
           "peak_gb": mx.get_peak_memory() / 1e9}
    del model
    gc.collect()
    mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--skip-mlx", action="store_true")
    args = ap.parse_args()
    tok = load_tokenizer(args.model)
    prompts = [render(tok, s, u, TOOLS).ids for s, u in smoke_prompts()]
    report = {"model": args.model, "revision": pins()[args.model]["revision"], "prompt_tokens": len(prompts[0]),
              "passes": args.passes, **torch_modes(args.model, prompts, args.passes, tok.convert_tokens_to_ids("<|im_end|>"))}
    if not args.skip_mlx:
        report["mlx"] = mlx_mode(args.model, prompts, args.passes)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"throughput_{args.model}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
