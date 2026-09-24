import hashlib
import urllib.request

import opharm
import yaml
from huggingface_hub import HfApi, hf_hub_download

from opharm.paths import CONFIGS, DATA, ROOT

HARMBENCH = ("centerforaisafety/HarmBench", "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
             "data/behavior_datasets/harmbench_behaviors_text_all.csv")
ALPACA = "tatsu-lab/alpaca"
PINS = CONFIGS / "data.yaml"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pins = yaml.safe_load(PINS.read_text()) if PINS.exists() else {}
    repo, commit, path = HARMBENCH
    dst = DATA / "raw" / "harmbench" / path.rsplit("/", 1)[-1]
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", dst)
    pins["harmbench"] = {"repo": repo, "commit": commit, "path": path, "local": str(dst.relative_to(ROOT)),
                         "sha256": sha256(dst), "license": "MIT"}
    if "alpaca" not in pins:
        info = HfApi().dataset_info(ALPACA)
        pins["alpaca"] = {"repo": ALPACA, "revision": info.sha, "license": info.card_data.get("license"),
                          "files": [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]}
    a = pins["alpaca"]
    for f in a["files"]:
        hf_hub_download(a["repo"], f, repo_type="dataset", revision=a["revision"])
    CONFIGS.mkdir(exist_ok=True)
    PINS.write_text(yaml.safe_dump(pins, sort_keys=False))
    print(yaml.safe_dump(pins, sort_keys=False))


if __name__ == "__main__":
    main()
