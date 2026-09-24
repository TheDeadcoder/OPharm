import argparse
import json

import opharm
import numpy as np
import torch

from opharm.bench.smoke import smoke_prompts
from opharm.bench.tools import TOOLS
from opharm.chat import load_tokenizer, pins, render
from opharm.interp import hooks
from opharm.models import load_model
from opharm.paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    tok = load_tokenizer(args.model)
    model = load_model(args.model)
    mods = hooks.resid_modules(model)
    attn = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
    stats = {k: [[] for _ in mods] for k in ("median_norm", "sink_norm", "inst_norm", "post_norm", "inst_max", "post_max")}
    top_dims = {"inst": [[] for _ in mods], "post": [[] for _ in mods]}

    for s, u in smoke_prompts():
        r = render(tok, s, u, TOOLS)
        store = {}

        def make(p):
            def fn(h):
                store[p] = h[0].float()
                return h
            return fn

        with torch.inference_mode(), hooks.hooked([("pre", m, make(p)) for p, m in enumerate(mods)]):
            model(input_ids=torch.tensor([r.ids], device=model.device), use_cache=False, logits_to_keep=1)
        for p, h in store.items():
            norms = h.norm(dim=-1)
            stats["median_norm"][p].append(norms[1:].median().item())
            stats["sink_norm"][p].append(norms[0].item())
            for name, t in (("inst", r.t_inst), ("post", r.t_post)):
                stats[f"{name}_norm"][p].append(norms[t].item())
                stats[f"{name}_max"][p].append(h[t].abs().max().item() / h[t].abs().median().item())
                top_dims[name][p].append(int(h[t].abs().argmax()))

    table = []
    for p in range(len(mods)):
        row = {"point": p, "block_in": "attention" if p in attn else ("final" if p == len(mods) - 1 else "gdn")}
        row.update({k: float(np.median(v[p])) for k, v in stats.items()})
        row.update({f"{k}_top_dim": max(set(v[p]), key=v[p].count) for k, v in top_dims.items()})
        table.append(row)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"resid_profile_{args.model}.json").write_text(json.dumps(
        {"model": args.model, "revision": pins()[args.model]["revision"], "attention_layers": attn, "rows": table}, indent=1))
    print("point block median_norm sink_norm inst_norm post_norm inst_max/med post_max/med")
    for row in table:
        print(f"{row['point']:>5} {row['block_in']:>9} {row['median_norm']:>11.1f} {row['sink_norm']:>9.1f} "
              f"{row['inst_norm']:>9.1f} {row['post_norm']:>9.1f} {row['inst_max']:>12.1f} {row['post_max']:>12.1f}")


if __name__ == "__main__":
    main()
