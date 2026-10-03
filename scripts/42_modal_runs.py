import argparse
import contextlib
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import modal

APP = "opharm-runs"
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HF = "/root/.cache/huggingface"
GPU, CPU, MEM = "H100!", 2.0, 32768
GPU_USD_S = 0.001097 + CPU * 0.0000131 + MEM / 1024 * 0.00000222
CPU_USD_S = 2.0 * 0.0000131 + 4.0 * 0.00000222
CAP_USD = float(os.environ.get("OPHARM_CAP_USD", "28"))
GATED = os.environ.get("OPHARM_GATED", "")
GATE = ["docs/assist_probe.md", "src/opharm/bench/assist.py", "scripts/41_assist_build.py", "tests/test_assist.py",
        "results/assist_instances.json", "scripts/42_modal_runs.py", "scripts/43_assist_report.py", "scripts/09_run_behavior.py",
        "scripts/24_continue.py"]
EXTRA_PINS = json.loads(os.environ.get("OPHARM_EXTRA_PINS", "{}"))
PINS = ["torch==2.14.0", "transformers==5.17.0", "tokenizers==0.23.2", "huggingface-hub==1.33.0", "hf-xet==1.6.0",
        "safetensors==0.8.0", "numpy==2.5.3", "scikit-learn==1.9.1", "scipy==1.18.1", "PyYAML==6.0.3", "Jinja2==3.1.6",
        "accelerate==1.15.0", "sentencepiece==0.2.2", "protobuf==7.36.2", "regex==2026.9.10", "pandas==3.0.6",
        "pillow==12.3.0", "psutil==7.2.2"]


def git_here():
    if os.environ.get("OPHARM_GIT") is not None:
        return os.environ["OPHARM_GIT"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *GATE], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .env({"PYTHONPATH": "/root/src", "OPHARM_GATED": GATED, "OPHARM_GIT": git_here(),
               "OPHARM_EXTRA_PINS": json.dumps(EXTRA_PINS)})
         .add_local_dir(ROOT / "src" / "opharm", "/root/src/opharm", ignore=["**/__pycache__/**"])
         .add_local_file(ROOT / "scripts" / "09_run_behavior.py", "/root/scripts/09_run_behavior.py")
         .add_local_file(ROOT / "scripts" / "24_continue.py", "/root/scripts/24_continue.py")
         .add_local_dir(ROOT / "configs", "/root/configs")
         .add_local_file(ROOT / "benchmark" / "manifest.json", "/root/benchmark/manifest.json")
         .add_local_dir(ROOT / "benchmark" / "assist", "/root/benchmark/assist"))
hf = modal.Volume.from_name("opharm-hf", create_if_missing=True)
runs = modal.Volume.from_name("opharm-runs", create_if_missing=True)
progress = modal.Dict.from_name("opharm-runs-progress", create_if_missing=True)
app = modal.App(APP, image=image)


def prepare():
    sys.path[:0] = ["/root/src", "/root/scripts"]
    import importlib
    import opharm
    import opharm.chat as oc
    import opharm.models as om
    if EXTRA_PINS:
        base = oc.pins
        oc.pins = om.pins = lambda: {**base(), **EXTRA_PINS}
    orig = om.load_model
    om.load_model = lambda key, device="mps", dtype=None, solver=None, local_files_only=True: orig(
        key, device="cuda", solver="loop", local_files_only=local_files_only)
    runner = importlib.import_module("09_run_behavior")
    runner.git_state = lambda: os.environ["OPHARM_GIT"]
    return runner, importlib.import_module("24_continue")


def fetch_weights(model, gated=False):
    sys.path.insert(0, "/root/src")
    import opharm
    import opharm.chat as oc
    from huggingface_hub import snapshot_download, whoami
    t0 = time.time()
    if gated and whoami().get("auth", {}).get("accessToken", {}).get("role") != "read":
        progress[f"download:{model}"] = {"state": "refused, token is not read-only", "updated": t0}
        raise RuntimeError("the Hugging Face secret does not hold a read token")
    p = {**oc.pins(), **EXTRA_PINS}[model]
    progress[f"download:{model}"] = {"state": "running", "updated": t0}
    path = Path(snapshot_download(p["repo"], revision=p["revision"],
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"]))
    hf.commit()
    s = time.time() - t0
    gb = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9, 2)
    progress[f"download:{model}"] = {"state": "done", "gb": gb, "billed_seconds": round(s), "usd": round(s * CPU_USD_S, 4),
                                     "updated": time.time()}
    return {"model": model, "gb": gb}


@app.function(cpu=2.0, memory=4096, timeout=3600, volumes={HF: hf})
def download(model):
    return fetch_weights(model)


if GATED:
    @app.function(cpu=2.0, memory=4096, timeout=3600, volumes={HF: hf}, secrets=[modal.Secret.from_name("opharm-hf-read")])
    def download_gated(model):
        return fetch_weights(model, gated=True)


def lines(path):
    return sum(1 for _ in open(path)) if path.exists() else 0


class TimeLimit(Exception):
    pass


def guarded(fn, limit):
    def call(*args, **kwargs):
        if limit.is_set():
            raise TimeLimit
        return fn(*args, **kwargs)
    return call


def watch(key, out, t0, prior, max_seconds, stop, limit):
    while not stop.wait(60):
        billed = prior + time.time() - t0
        if billed > max_seconds:
            limit.set()
        progress[key] = {**progress.get(key, {}), "state": "stopping" if limit.is_set() else "running", "billed_seconds": round(billed, 1),
                         "usd": round(billed * GPU_USD_S, 3), "prefill": lines(out / "prefill.jsonl"),
                         "decode": lines(out / "decode.jsonl"), "judge": lines(out / "judge.jsonl"), "updated": time.time()}
        runs.commit()


@app.function(gpu=GPU, cpu=CPU, memory=MEM, timeout=4 * 3600, volumes={HF: hf, "/root/runs": runs})
def behave(model, tag, argv, cont, max_seconds, run_id):
    t0 = time.time()
    runner, cont_mod = prepare()
    import gc
    import torch
    from opharm.run.cache import forward_capture
    from opharm.run.generate import greedy
    gc.collect()
    torch.cuda.empty_cache()
    key = f"{model}:{tag}"
    prev = progress.get(key, {})
    same = prev.get("run_id") == run_id
    prior, attempt = (prev.get("billed_seconds", 0.0), prev.get("attempt", 0) + 1) if same else (0.0, 1)
    progress[key] = {"run_id": run_id, "attempt": attempt, "state": "running", "billed_seconds": prior, "usd": round(prior * GPU_USD_S, 3),
                     "devices": [*(prev.get("devices", []) if same else []), torch.cuda.get_device_name(0)], "updated": time.time()}
    out = Path("/root/runs") / model / tag
    stop, limit = threading.Event(), threading.Event()
    runner.greedy, runner.forward_capture = guarded(greedy, limit), guarded(forward_capture, limit)
    watcher = threading.Thread(target=watch, args=(key, out, t0, prior, max_seconds, stop, limit), daemon=True)
    watcher.start()
    capped, state = 0, "failed"
    try:
        sys.argv = ["09_run_behavior.py", model, "--tag", tag, *argv]
        runner.main()
        if cont:
            buf = io.StringIO()
            sys.argv = ["24_continue.py", model, "--tag", tag, *cont, "--dry-run"]
            with contextlib.redirect_stdout(buf):
                cont_mod.main()
            capped = json.loads(buf.getvalue().strip().splitlines()[-1])["capped"]
            if capped:
                sys.argv = ["24_continue.py", model, "--tag", tag, *cont, "--batch", "8"]
                cont_mod.main()
        state = "done"
    except TimeLimit:
        state = "stopped_time"
    finally:
        stop.set()
        watcher.join(120)
        runs.commit()
        billed = prior + time.time() - t0
        progress[key] = {**progress.get(key, {}), "state": state, "billed_seconds": round(billed, 1), "usd": round(billed * GPU_USD_S, 3),
                         "prefill": lines(out / "prefill.jsonl"), "decode": lines(out / "decode.jsonl"), "judge": lines(out / "judge.jsonl"),
                         "capped": capped, "updated": time.time()}
    return {"model": model, "tag": tag, "state": state, "capped": capped, "seconds": round(billed)}


def spent():
    total = 0.0
    for name in ("opharm-progress", "opharm-runs-progress"):
        try:
            total += sum(v.get("usd", 0.0) for _, v in modal.Dict.from_name(name).items())
        except modal.exception.NotFoundError:
            pass
    return total


def gate():
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *GATE], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refused: commit these first:\n{dirty}")
    info = json.loads((ROOT / "results" / "assist_instances.json").read_text())
    for name, f in info["files"].items():
        if hashlib.sha256((ROOT / "benchmark" / "assist" / f"{name}.jsonl").read_bytes()).hexdigest() != f["sha256"]:
            raise SystemExit(f"refused: benchmark/assist/{name}.jsonl differs from results/assist_instances.json")


def launch(fn, model, tag, argv, cont, max_minutes, run_id):
    if fn == "behave":
        if not tag.startswith("assist_pilot"):
            gate()
        if max_minutes > 230:
            raise SystemExit("refused: the time limit must end before the 4-hour function timeout")
        if spent() + max_minutes * 60 * GPU_USD_S > CAP_USD:
            raise SystemExit(f"refused: ${spent():.2f} spent, this call could add ${max_minutes * 60 * GPU_USD_S:.2f}, cap ${CAP_USD:.0f}")
    f = modal.Function.from_name(APP, fn)
    call = f.spawn(model) if fn != "behave" else f.spawn(model, tag, argv, cont, max_minutes * 60, run_id)
    rec = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "fn": fn, "model": model, "tag": tag, "argv": argv, "cont": cont,
           "max_minutes": max_minutes, "run_id": run_id, "call": call.object_id}
    (RUNS / "modal").mkdir(exist_ok=True)
    with open(RUNS / "modal" / "calls.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def fetch(model, tag):
    for t in (tag, tag + "_cont"):
        try:
            entries = runs.listdir(f"{model}/{t}")
        except Exception:
            continue
        dest = RUNS / model / t
        if (dest / "results.jsonl").exists():
            raise FileExistsError(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for e in entries:
            name = Path(e.path).name
            if name != "acts.npy":
                (dest / name).write_bytes(b"".join(runs.read_file(e.path)))
        print(f"fetched {dest}")


def status():
    now = time.time()
    for k, v in sorted(progress.items()):
        age = round((now - v.get("updated", now)) / 60, 1)
        stale = " STALE?" if v.get("state") in ("running", "stopping") and age > 5 else ""
        print(f"{k:34s} {v.get('state', ''):13s} prefill {v.get('prefill', '-')} decode {v.get('decode', '-')} judge {v.get('judge', '-')}  "
              f"{v.get('billed_seconds', 0) / 60:6.1f} min ${v.get('usd', 0):6.3f}  updated {age} min ago{stale}")
    print(f"spent about ${spent():.2f} of the ${CAP_USD:.0f} cap on this account")


def cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("fn", choices=["download", "download_gated", "behave"])
    p.add_argument("model")
    p.add_argument("--tag", default="")
    p.add_argument("--max-minutes", type=float, default=30)
    p.add_argument("--run-id", default="")
    p.add_argument("--args", default="")
    p.add_argument("--cont", default="")
    sub.add_parser("status")
    p = sub.add_parser("fetch")
    p.add_argument("model")
    p.add_argument("tag")
    a = ap.parse_args()
    if a.cmd == "launch":
        launch(a.fn, a.model, a.tag, shlex.split(a.args), shlex.split(a.cont), a.max_minutes,
               a.run_id or f"{datetime.now():%Y%m%d-%H%M%S}")
    elif a.cmd == "status":
        status()
    else:
        fetch(a.model, a.tag)


if __name__ == "__main__":
    cli()
