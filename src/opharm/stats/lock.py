import hashlib

import yaml

from opharm.paths import CONFIGS, ROOT

LOCK = ROOT / "prereg.lock"
SEALED_BY = (ROOT / "docs" / "prereg.md", ROOT / "docs" / "prereg_amendment_1.md", ROOT / "docs" / "prereg_amendment_2.md",
             CONFIGS / "locked.yaml")


def prereg_hash():
    h = hashlib.sha256()
    for p in SEALED_BY:
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def unlocked():
    return LOCK.exists() and LOCK.read_text().strip() == prereg_hash()


def dev_only(rows):
    return [r for r in rows if r["split"] == "dev"]


def locked_models():
    path = CONFIGS / "locked.yaml"
    return set(yaml.safe_load(path.read_text()) or {}) if path.exists() else set()


def analysis_rows(rows, model=None):
    if any(r["split"] == "heldout" for r in rows):
        if not unlocked():
            raise PermissionError("held-out rows are sealed until prereg.lock matches the registered documents")
        if model is not None and model not in locked_models():
            raise PermissionError(f"held-out rows of {model} are sealed until its values are in configs/locked.yaml")
    return rows
