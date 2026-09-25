import hashlib
import json

import numpy as np

from opharm.paths import RUNS


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def half(skeleton):
    return int(hashlib.sha256(skeleton.encode()).hexdigest(), 16) % 2


def unmatched(x, pos, neg, skeletons):
    halves = np.array([half(s) for s in skeletons])
    parts = [x[pos & (halves == h)].mean(0) - x[neg & (halves != h)].mean(0) for h in (0, 1)]
    return 0.5 * (parts[0] + parts[1])


def position(x, mu_benign, mu_harm):
    u = unit(mu_harm - mu_benign)
    return ((x - mu_benign) * u).sum(-1) / ((mu_harm - mu_benign) * u).sum(-1)


def content_centroids(model):
    run = RUNS / model / "refsets"
    items = [json.loads(line) for line in open(run / "items.jsonl")]
    acts = np.load(run / "acts.npy", mmap_mode="r")
    ext = np.array([it["split"] == "extract" for it in items])
    harm = ext & np.array([it["kind"] == "harmful" and it["refused"] for it in items])
    ok = ext & np.array([it["kind"] == "benign" and not it["refused"] for it in items])
    return {"harm": acts[harm].mean(0), "benign": acts[ok].mean(0)}
