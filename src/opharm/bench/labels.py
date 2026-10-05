import hashlib
import json
from functools import lru_cache

from opharm.bench.evaluate import evaluate
from opharm.chat import load_tokenizer
from opharm.paths import RUNS


@lru_cache(maxsize=None)
def tokenizer(model):
    return load_tokenizer(model)


@lru_cache(maxsize=None)
def run_max_new(model, tag):
    path = RUNS / model / tag / "manifest.json"
    return json.loads(path.read_text())["args"]["max_new"] if path.exists() else None


def capped(text, model, max_new):
    return bool(max_new) and len(tokenizer(model).encode(text, add_special_tokens=False)) >= max_new - 1


def label(text, row, model, max_new=None, strict=False):
    return evaluate(text, row, model, truncated=capped(text, model, max_new) if max_new else None, strict=strict)


@lru_cache(maxsize=None)
def continuation(model, tag):
    path = RUNS / model / f"{tag}_cont"
    if not (path / "results.jsonl").exists():
        return {}, 0
    manifest = json.loads((path / "manifest.json").read_text())
    if hashlib.sha256((RUNS / model / tag / "results.jsonl").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError(f"{model}/{tag}: results changed after the continuation")
    return {r["id"]: r for r in map(json.loads, open(path / "results.jsonl"))}, manifest["args"]["max_more"]


def final_label(row, model, tag, strict=False):
    cont, extra = continuation(model, tag)
    base = run_max_new(model, tag)
    c = cont.get(row["id"])
    if c is None:
        return label(row["text"], row, model, base, strict)
    return label(c["text"], row, model, base + extra if c["capped_again"] else None, strict)
