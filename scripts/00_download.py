import argparse
import os

import opharm
from opharm.paths import CONFIGS

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import yaml
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

MODELS = {
    "qwen35_08b": ("Qwen/Qwen3.5-0.8B", "harness"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "dev"),
    "qwen3_4b_2507": ("Qwen/Qwen3-4B-Instruct-2507", "comparator"),
    "qwen35_9b": ("Qwen/Qwen3.5-9B", "confirmatory"),
    "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", "comparator_2"),
}
DEFAULT = ["qwen35_08b", "qwen35_4b", "qwen3_4b_2507", "qwen35_9b"]
SMALL = ["*.json", "*.jinja", "*.txt", "LICENSE*", "README.md"]
PINS = CONFIGS / "models.yaml"


def pin(api, keys):
    pins = yaml.safe_load(PINS.read_text()) if PINS.exists() else {}
    for key in keys:
        if key in pins:
            continue
        repo, role = MODELS[key]
        info = api.model_info(repo, files_metadata=True)
        size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
        pins[key] = {"repo": repo, "revision": info.sha, "role": role, "safetensors_bytes": size}
    CONFIGS.mkdir(exist_ok=True)
    PINS.write_text(yaml.safe_dump(pins, sort_keys=False))
    return pins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=DEFAULT)
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    pins = pin(HfApi(), args.keys)
    for key in args.keys:
        p = pins[key]
        try:
            path = snapshot_download(p["repo"], revision=p["revision"], allow_patterns=SMALL if args.small else None)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"{key}: no access ({type(e).__name__})")
            continue
        print(f"{key}: {'small' if args.small else 'full'} {p['revision'][:12]} {p['safetensors_bytes'] / 1e9:.2f} GB -> {path}")


if __name__ == "__main__":
    main()
