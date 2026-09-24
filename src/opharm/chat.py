import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from opharm.paths import CONFIGS


def pins():
    return yaml.safe_load((CONFIGS / "models.yaml").read_text())


def snapshot_dir(key):
    p = pins()[key]
    return Path(os.environ["HF_HUB_CACHE"]) / f"models--{p['repo'].replace('/', '--')}" / "snapshots" / p["revision"]


def load_tokenizer(key, local_files_only=True):
    p = pins()[key]
    return AutoTokenizer.from_pretrained(p["repo"], revision=p["revision"], local_files_only=local_files_only)


@dataclass
class Rendered:
    text: str
    ids: list
    offsets: list
    spans: dict
    t_inst: int
    t_post: int


def _locate(text, content):
    if text.count(content) != 1:
        raise ValueError("segment text must occur exactly once in the rendered prompt")
    return text.index(content)


def _token_span(offsets, start, end):
    hit = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    return hit[0], hit[-1] + 1


def render(tok, system, user, tools=None, thinking=False):
    sys_text = "".join(t for _, t in system)
    user_text = "".join(t for _, t in user)
    if any(c != c.strip() for c in (sys_text, user_text)):
        raise ValueError("message content must not start or end with whitespace, some templates trim it")
    msgs =[{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}]
    text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    spans = {}
    for segments, content in ((system, sys_text), (user, user_text)):
        pos = _locate(text, content)
        for name, seg in segments:
            if name:
                spans.setdefault(name, []).append(_token_span(offsets, pos, pos + len(seg)))
            pos += len(seg)
    user_end = _locate(text, user_text) + len(user_text)
    t_inst = max(i for i, (a, _) in enumerate(offsets) if a < user_end)
    return Rendered(text, ids, offsets, spans, t_inst, len(ids) - 1)
