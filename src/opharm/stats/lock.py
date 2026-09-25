import hashlib

from opharm.paths import CONFIGS, ROOT

LOCK = ROOT / "prereg.lock"
SEALED_BY = (ROOT / "docs" / "prereg.md", ROOT / "docs" / "prereg_amendment_1.md", CONFIGS / "locked.yaml")


def prereg_hash():
    h = hashlib.sha256()
    for p in SEALED_BY:
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def unlocked():
    return LOCK.exists() and LOCK.read_text().strip() == prereg_hash()


def dev_only(rows):
    return [r for r in rows if r["split"] == "dev"]


def analysis_rows(rows):
    if any(r["split"] == "heldout" for r in rows) and not unlocked():
        raise PermissionError("held-out rows are sealed until prereg.lock matches docs/prereg.md and configs/locked.yaml")
    return rows
