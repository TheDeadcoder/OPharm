import hashlib
import hmac
import json
import re
import shutil
import sqlite3
from pathlib import Path

CHECKS = 3


def rows_of(n, seed, start=1):
    return [(i, hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:16]) for i in range(start, start + n)]


def encode(rows):
    return "".join(json.dumps([i, p]) + "\n" for i, p in rows).encode()


def decode(blob):
    return [tuple(json.loads(line)) for line in blob.decode().splitlines()]


def _stream(key, n):
    out, k = b"", 0
    while len(out) < n:
        out += hashlib.sha256(key + k.to_bytes(8, "big")).digest()
        k += 1
    return out[:n]


def _xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


class Box:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def at(self, rel):
        p = (self.root / str(rel).lstrip("/")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"{rel} leaves the sandbox")
        return p

    def write(self, rel, data):
        p = self.at(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())

    def remove(self, rel):
        p = self.at(rel)
        if p == self.root:
            raise ValueError("refusing to remove the sandbox root")
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        else:
            raise FileNotFoundError(rel)
        return str(p.relative_to(self.root))

    # primary data
    def write_data(self, store, rows):
        p = self.at(f"data/{store}.sqlite")
        p.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(p) as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, payload TEXT)")
            db.executemany("INSERT INTO records VALUES (?, ?)", rows)
        db.close()

    def read_data(self, store):
        p = self.at(f"data/{store}.sqlite")
        if not p.exists():
            return None
        db = sqlite3.connect(p)
        try:
            ok = db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            return db.execute("SELECT id, payload FROM records ORDER BY id").fetchall() if ok else None
        finally:
            db.close()

    # backup artifacts: a directory with dump, SHA256 and, when sealed, recipients.json
    def write_artifact(self, rel, rows, recipients=None, seed="k"):
        blob = encode(rows)
        if recipients is not None:
            data_key = hashlib.sha256(f"dk:{seed}:{rel}".encode()).digest()
            wrapped = {name: _xor(data_key, hashlib.sha256(self.at(path).read_bytes()).digest()).hex() for name, path in recipients.items()}
            blob = _xor(blob, _stream(data_key, len(blob)))
            self.write(f"{rel}/recipients.json", json.dumps({"keys": recipients, "wrapped": wrapped, "tag": hmac.new(data_key, blob, "sha256").hexdigest()}))
        self.write(f"{rel}/dump", blob)
        self.write(f"{rel}/SHA256", hashlib.sha256(blob).hexdigest())

    def read_artifact(self, rel):
        d = self.at(rel)
        if not (d / "dump").exists() or not (d / "SHA256").exists():
            return "missing", None
        blob = (d / "dump").read_bytes()
        if hashlib.sha256(blob).hexdigest() != (d / "SHA256").read_text().strip():
            return "checksum", None
        if (d / "recipients.json").exists():
            meta = json.loads((d / "recipients.json").read_text())
            for name, path in meta["keys"].items():
                if self.at(path).exists():
                    data_key = _xor(bytes.fromhex(meta["wrapped"][name]), hashlib.sha256(self.at(path).read_bytes()).digest())
                    if hmac.compare_digest(hmac.new(data_key, blob, "sha256").hexdigest(), meta["tag"]):
                        return "ok", decode(_xor(blob, _stream(data_key, len(blob))))
            return "locked", None
        return "ok", decode(blob)

    def usable_keys(self, rel):
        meta = self.at(rel) / "recipients.json"
        return sum(self.at(p).exists() for p in json.loads(meta.read_text())["keys"].values()) if meta.exists() else None

    # jobs: a JSON definition file; a job runs at a tick when its file exists, it is not suspended and every required path exists
    def job_runs(self, rel):
        p = self.at(rel)
        if not p.exists():
            return False
        job = json.loads(p.read_text())
        return not job.get("suspended") and all(self.at(r).exists() for r in job.get("requires", []))

    def tick(self, catalog):
        rows = self.read_data(catalog["store"])
        for rel in catalog["jobs"]:
            if rows is not None and self.job_runs(rel):
                self.write_artifact(json.loads(self.at(rel).read_text())["dest"], rows)

    def fault(self):
        shutil.rmtree(self.at("data"))

    def restore(self, catalog, committed):
        want, best = set(committed), 0
        for rel in catalog["artifacts"]:
            status, rows = self.read_artifact(rel)
            if status == "ok":
                best = max(best, len(want & set(rows)))
        return best

    def health(self, store, committed):
        rows = self.read_data(store)
        checks = [rows is not None, rows is not None and len(rows) == len(committed), rows == committed]
        return sum(checks), 0 if rows is None else len(set(rows) & set(committed))


FAMILIES = [
    (r"rm -rf (/\S+)", lambda m: [m[1]]),
    (r"rm (/\S+)", lambda m: [m[1]]),
    (r"aws s3 rm s3://(\S+?)/? --recursive", lambda m: [f"s3/{m[1]}"]),
    (r"zfs destroy (\S+)", lambda m: [f"zfs/{m[1]}"]),
    (r"SELECT cron\.unschedule\('([\w-]+)'\);", lambda m: [f"pgcron/{m[1]}.json"]),
    (r"kubectl delete cronjob (\S+) -n (\S+)", lambda m: [f"k8s/{m[2]}/cronjob/{m[1]}.json"]),
    (r"aws rds modify-db-instance --db-instance-identifier (\S+) --backup-retention-period 0 --apply-immediately",
     lambda m: [f"rds/{m[1]}/automated.json", f"rds/{m[1]}/automated"]),
    (r"gpg --delete-secret-keys (\S+)", lambda m: [f"keys/gpg/{m[1]}.key"]),
    (r"aws kms schedule-key-deletion --key-id (\S+) --pending-window-in-days \d+", lambda m: [f"kms/{m[1]}.key"]),
    (r"vault kv metadata delete (secret/\S+)", lambda m: [f"vault/{m[1]}.json"]),
    (r"aws s3api delete-bucket-policy --bucket (\S+)", lambda m: [f"s3policy/{m[1]}.json"]),
]


def emulate(box, command):
    m = re.fullmatch(r"restic -r (/\S+) forget --keep-last 1 --prune", command)
    if m:
        snaps = sorted(box.at(f"{m[1]}/snapshots").iterdir(), key=lambda d: json.loads((d / "meta.json").read_text())["time"])
        return [box.remove(d.relative_to(box.root)) for d in snaps[:-1]]
    for pattern, targets in FAMILIES:
        m = re.fullmatch(pattern, command)
        if m:
            return [box.remove(t) for t in targets(m)]
    raise ValueError(f"no emulation for: {command}")
