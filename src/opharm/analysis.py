import json

import numpy as np
import yaml

from opharm.paths import BENCH, CONFIGS, RESULTS, RUNS
from opharm.stats.lock import analysis_rows, dev_only

TRAIN_CLASSES, TRAIN_FORMS = ("file", "database", "compute", "access"), ("F1", "F2")
POS = {"t_inst": 0, "t_post": 1}


def code(r):
    return r["target"] + r["env"] + r["rollback"] + r["policy"]


def user_text(r):
    return "".join(t for _, t in r["user"])


def eval_split(confirm):
    return "heldout" if confirm else "dev"


def load_rows(model, tag, confirm):
    run = RUNS / model / tag
    meta = {m["id"]: m for m in map(json.loads, open(BENCH / "instances.jsonl"))}
    rows = [dict(meta[r["id"]], **{**r, "row": k}) for k, r in enumerate(map(json.loads, open(run / "results.jsonl")))]
    return (analysis_rows(rows) if confirm else dev_only(rows)), np.load(run / "acts.npy", mmap_mode="r")


def blast_partition(rows, confirm):
    train = np.array([r["split"] == "dev" and r["cls"] in TRAIN_CLASSES and r["form"] in TRAIN_FORMS for r in rows])
    test = np.array([r["split"] == eval_split(confirm) and r["cls"] not in TRAIN_CLASSES and r["form"] not in TRAIN_FORMS
                     for r in rows])
    forms = np.array([r["form"] for r in rows])[train]
    splits = [(np.flatnonzero(forms == a), np.flatnonzero(forms == b)) for a, b in (TRAIN_FORMS, TRAIN_FORMS[::-1])]
    return train, test, splits


def locked(model):
    return yaml.safe_load((CONFIGS / "locked.yaml").read_text())[model]


def settings(model, tag, confirm):
    if confirm:
        return locked(model)
    ref = json.loads((RESULTS / f"refsets_{model}.json").read_text())["auroc_eval"]
    rep = json.loads((RESULTS / f"representation_{model}_{tag}.json").read_text())["represented"]
    return {"L_harm": int(np.argmax(ref["r_harm"])), "L_ref": int(np.argmax(ref["r_ref"])),
            "L_steer": json.loads((RESULTS / f"c5_{model}.json").read_text())["selected_layer"], "steer_coef": 1.0,
            "blast": {name: [d["position"], d["layer"]] for name, d in rep["points"].items()}}
