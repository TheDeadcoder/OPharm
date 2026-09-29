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
