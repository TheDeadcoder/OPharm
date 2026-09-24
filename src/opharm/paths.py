import os
import sys
from pathlib import Path

_EARLY = [m for m in ("huggingface_hub", "transformers", "mlx_lm", "matplotlib") if m in sys.modules]
if _EARLY:
    raise ImportError(f"import opharm before {', '.join(_EARLY)} so every cache stays inside the project")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
BENCH = ROOT / "benchmark"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

_LOCAL = {
    "HF_HOME": CACHE / "huggingface",
    "HF_HUB_CACHE": CACHE / "huggingface" / "hub",
    "HF_XET_CACHE": CACHE / "huggingface" / "xet",
    "TORCH_HOME": CACHE / "torch",
    "MPLCONFIGDIR": CACHE / "matplotlib",
    "XDG_CACHE_HOME": CACHE,
}
for _k, _v in _LOCAL.items():
    os.environ[_k] = str(_v)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _dotenv(path):
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.removeprefix("export ").split("=", 1)
                out[key.strip()] = value.strip().strip("'\"")
    return out


if not os.environ.get("HF_TOKEN"):
    _token = _dotenv(ROOT / ".env").get("HF_READ")
    if _token:
        os.environ["HF_TOKEN"] = _token
