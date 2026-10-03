import argparse
import json
import random

import opharm
import numpy as np

from opharm.analysis import POS, code, load_rows, settings
from opharm.interp.directions import unmatched
from opharm.paths import RUNS

KEEP = ("id", "skeleton", "set", "split", "cls", "template", "form", "target", "env", "rollback", "policy", "system", "user")
OUT = RUNS / "modal" / "bundles"


def dev_main(rows):
    return [r for r in rows if r["set"] == "main" and r["split"] == "dev"]


def blast_dirs(rows, acts):
    dm = dev_main(rows)
    x = np.asarray(acts[[r["row"] for r in dm]])
    env = np.array([r["env"] == "P" for r in dm])
    skel = np.array([r["skeleton"] for r in dm])
    return {p: unmatched(x[:, i], env, ~env, skel) for p, i in POS.items()}


def sample(main, groups, n):
    rng = random.Random(2)
    rows = []
    for g in groups:
        cand = [r for r in main if code(r) in {g + "AN", g + "NN"}]
        rows += rng.sample(cand, min(n, len(cand)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--parity-rows", type=int, default=3)
    args = ap.parse_args()
    m = args.model
    rows, acts = load_rows(m, "grid", False)
    st = settings(m, "grid", False)
    ladder = sample([r for r in rows if r["set"] == "main" and r["split"] == "dev"], ["DP", "DS", "BP"], 60)
    rb = blast_dirs(rows, acts)
    rows_c, acts_c = load_rows(m, "grid", True)
    same = [r["row"] for r in dev_main(rows_c)] == [r["row"] for r in dev_main(rows)]
    rb_c = rb if same else blast_dirs(rows_c, acts_c)
    st_c = settings(m, "grid", True)
    parity = sample([r for r in rows_c if r["set"] == "main" and r["split"] == "heldout"], ["DP"], 1000)[: args.parity_rows]
    mps = [json.loads(line) for line in open(RUNS / m / "grid" / f"causal_c3_{m}_grid_confirm_h4.jsonl")]
    if [r["id"] for r in parity] != list(dict.fromkeys(x["id"] for x in mps))[: args.parity_rows]:
        raise ValueError("parity rows do not match the MPS h4 run")
    trim = lambda rs: [{k: r[k] for k in KEEP} for r in rs]
    doc = {"c3": {"name": f"causal_c3_{m}_grid", "rows": trim(ladder), "coefs": [0.1, 0.25, 0.5, 1.0, 2.0], "seeds": 8,
                  "L_steer": st["L_steer"], "blast": st["blast"]},
           "parity": {"name": f"parity_c3_{m}_grid_confirm_h4", "rows": trim(parity), "coefs": [1.0], "seeds": 24,
                      "L_steer": st_c["L_steer"], "blast": st_c["blast"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{m}.json").write_text(json.dumps(doc))
    np.savez(OUT / f"{m}.npz", r_ref=np.load(RUNS / m / "refsets" / "directions.npz")["r_ref"],
             c3_t_inst=rb["t_inst"], c3_t_post=rb["t_post"], parity_t_inst=rb_c["t_inst"], parity_t_post=rb_c["t_post"])
    print(json.dumps({"model": m, "ladder_rows": len(ladder), "parity_rows": len(parity), "same_dev_rows": same,
                      "L_steer": [st["L_steer"], st_c["L_steer"]],
                      "bytes": {f.name: f.stat().st_size for f in sorted(OUT.glob(f"{m}.*"))}}))


if __name__ == "__main__":
    main()
