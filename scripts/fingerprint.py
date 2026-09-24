import hashlib
import json
import platform
import subprocess
from importlib.metadata import version

import opharm
from opharm.chat import load_tokenizer, pins, snapshot_dir
from opharm.paths import RESULTS

PACKAGES = ["torch", "transformers", "accelerate", "safetensors", "huggingface-hub", "hf-xet", "tokenizers",
            "numpy", "scipy", "scikit-learn", "pandas", "mlx", "mlx-lm"]
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_complete(d):
    index = d / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    return all((d / s).exists() for s in shards)


def main():
    out = {
        "python": platform.python_version(), "macos": platform.mac_ver()[0], "machine": platform.machine(),
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout),
        "packages": {p: version(p) for p in PACKAGES},
        "models": {},
    }
    for key, p in pins().items():
        d = snapshot_dir(key)
        out["models"][key] = {
            "repo": p["repo"], "revision": p["revision"], "weights_complete": weights_complete(d),
            "files_sha256": {f: sha256(d / f) for f in FILES if (d / f).exists()},
            "chat_template_sha256": hashlib.sha256(load_tokenizer(key).chat_template.encode()).hexdigest(),
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fingerprint.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v["weights_complete"], v["chat_template_sha256"][:12]) for k, v in out["models"].items()}))


if __name__ == "__main__":
    main()
