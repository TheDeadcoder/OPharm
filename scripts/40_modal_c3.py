import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

APP = "opharm-c3"
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HF = "/root/.cache/huggingface"
GPU, CPU, MEM = "H100", 2.0, 24576
GPU_USD_S = 0.001097 + CPU * 0.0000131 + MEM / 1024 * 0.00000222
CPU_USD_S = 2.0 * 0.0000131 + 4.0 * 0.00000222
CAP_USD = 25.0
GATED = os.environ.get("OPHARM_GATED", "")
PINS = ["torch==2.14.0", "transformers==5.17.0", "tokenizers==0.23.2", "huggingface-hub==1.33.0", "hf-xet==1.6.0",
        "safetensors==0.8.0", "numpy==2.5.3", "scikit-learn==1.9.1", "scipy==1.18.1", "PyYAML==6.0.3", "Jinja2==3.1.6",
        "accelerate==1.15.0", "sentencepiece==0.2.2", "protobuf==7.36.2", "regex==2026.9.10", "pandas==3.0.6",
        "pillow==12.3.0", "psutil==7.2.2"]

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .env({"PYTHONPATH": "/root/src", "OPHARM_GATED": GATED})
         .add_local_dir(ROOT / "src" / "opharm", "/root/src/opharm", ignore=["**/__pycache__/**"])
         .add_local_file(ROOT / "scripts" / "12_causal.py", "/root/scripts/12_causal.py")
         .add_local_dir(ROOT / "configs", "/root/configs")
         .add_local_dir(RUNS / "modal" / "bundles", "/root/bundles"))
hf = modal.Volume.from_name("opharm-hf", create_if_missing=True)
out = modal.Volume.from_name("opharm-out", create_if_missing=True)
progress = modal.Dict.from_name("opharm-progress", create_if_missing=True)
app = modal.App(APP, image=image)


def fetch_weights(model, gated=False):
    sys.path.insert(0, "/root/src")
    import opharm
    from huggingface_hub import snapshot_download, whoami
    from opharm.chat import pins
    t0 = time.time()
    if gated:
        role = whoami().get("auth", {}).get("accessToken", {}).get("role")
        if role != "read":
            progress[f"download:{model}"] = {"state": f"refused, token role {role}", "updated": t0}
            raise RuntimeError(f"the Hugging Face secret holds a {role} token, not a read token")
    p = pins()[model]
    progress[f"download:{model}"] = {"state": "running", "updated": t0}
    path = Path(snapshot_download(p["repo"], revision=p["revision"],
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"]))
    hf.commit()
    gb = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9, 2)
    s = time.time() - t0
    progress[f"download:{model}"] = {"state": "done", "gb": gb, "billed_seconds": round(s), "usd": round(s * CPU_USD_S, 4),
                                     "updated": time.time()}
    return {"model": model, "gb": gb, "seconds": round(s)}


@app.function(cpu=2.0, memory=4096, timeout=3600, volumes={HF: hf})
def download(model):
    return fetch_weights(model)


if GATED:
    @app.function(cpu=2.0, memory=4096, timeout=3600, volumes={HF: hf}, secrets=[modal.Secret.from_name("opharm-hf-read")])
    def download_gated(model):
        return fetch_weights(model, gated=True)


@app.function(gpu=GPU, cpu=CPU, memory=MEM, timeout=4 * 3600, volumes={HF: hf, "/out": out})
def ladder(model, mode, max_seconds, run_id):
    t0 = time.time()
    sys.path.insert(0, "/root/src")
    import importlib.util
    import opharm
    import numpy as np
    import torch
    import transformers
    key = f"{model}:{mode}"
    prev = progress.get(key, {})
    same = prev.get("run_id") == run_id
    prior, attempt = (prev.get("billed_seconds", 0.0), prev.get("attempt", 0) + 1) if same else (0.0, 1)
    spec = importlib.util.spec_from_file_location("c12", "/root/scripts/12_causal.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    b = json.loads(Path(f"/root/bundles/{model}.json").read_text())[mode]
    arr = np.load(f"/root/bundles/{model}.npz")
    ctx = C.Ctx.__new__(C.Ctx)
    ctx.key, ctx.tag = model, "grid"
    ctx.tok, ctx.model = C.load_tokenizer(model), C.load_model(model, device="cuda", solver="loop")
    ctx.inputs, ctx.limit = C.hooks.has_layer_inputs(ctx.model), C.hooks.last_patchable_layer(ctx.model)
    ctx.opener = C.opener_ids(ctx.tok)
    ctx.blast, ctx.l_steer = {k: tuple(v) for k, v in b["blast"].items()}, b["L_steer"]
    ctx.r_blast = {"t_inst": arr[f"{mode}_t_inst"], "t_post": arr[f"{mode}_t_post"]}
    ctx.r_ref, ctx.probes = arr["r_ref"], {}
    rows, per_row = b["rows"], len(b["coefs"]) * (4 + 2 * b["seeds"])
    path = Path("/out") / model / f"{b['name']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            have.setdefault(rec["id"], []).append(rec)
    done = 0
    while done < len(rows) and len(have.get(rows[done]["id"], [])) == per_row:
        done += 1
    path.write_text("".join(json.dumps(x) + "\n" for r in rows[:done] for x in have[r["id"]]))
    out.commit()
    device = torch.cuda.get_device_name(0)
    state, billed, t_loop, t_commit, start = "running", prior, time.time(), time.time(), done
    for i in range(done, len(rows)):
        recs = C.steer_panel(ctx, [rows[i]], b["coefs"], b["seeds"])
        with path.open("a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in recs))
        done, billed = i + 1, prior + time.time() - t0
        if time.time() - t_commit > 120 or done == len(rows):
            out.commit()
            t_commit = time.time()
        progress[key] = {"run_id": run_id, "attempt": attempt, "state": state, "rows_done": done, "rows_total": len(rows),
                         "billed_seconds": round(billed, 1), "usd": round(billed * GPU_USD_S, 3), "device": device,
                         "sec_per_forward": round((time.time() - t_loop) / ((done - start) * (per_row + 1)), 4),
                         "updated": time.time()}
        if billed > max_seconds and done < len(rows):
            state = "stopped_time"
            break
    out.commit()
    records = sum(1 for _ in path.open())
    complete = records == per_row * len(rows)
    n_layers = len(C.hooks.layers(ctx.model))
    summary = {"test": "c3", "records": records, "layers": sorted({round(1 + i * (n_layers - 3) / 7) for i in range(8)}),
               "blast": ctx.blast, "L_steer": ctx.l_steer, "seconds": round(billed, 1)}
    platform = {"device": device, "torch": torch.__version__, "transformers": transformers.__version__, "solver": "loop",
                "dtype": "bfloat16", "attempts": attempt, "complete": complete}
    (path.parent / f"{b['name']}.summary.json").write_text(json.dumps({"summary": summary, "platform": platform}))
    out.commit()
    billed = prior + time.time() - t0
    progress[key] = {**progress.get(key, {}), "state": "done" if complete else state, "rows_done": done,
                     "billed_seconds": round(billed, 1), "usd": round(billed * GPU_USD_S, 3), "updated": time.time()}
    return {"model": model, "mode": mode, "records": records, "complete": complete}


def spent():
    return sum(v.get("usd", 0.0) for _, v in progress.items())


def bundle(model, mode):
    return json.loads((RUNS / "modal" / "bundles" / f"{model}.json").read_text())[mode]


def fetch(model, mode):
    name = bundle(model, mode)["name"]
    dest = RUNS / "modal" / "out" / model
    dest.mkdir(parents=True, exist_ok=True)
    for fn in (f"{name}.jsonl", f"{name}.summary.json"):
        try:
            (dest / fn).write_bytes(b"".join(out.read_file(f"{model}/{fn}")))
        except Exception as e:
            print(f"{fn}: {type(e).__name__}")
    return dest / f"{name}.jsonl"


def parity(model):
    got = [json.loads(line) for line in open(fetch(model, "parity"))]
    mps = {(r["id"], r["direction"], r["coef"]): r for r in map(json.loads, open(RUNS / model / "grid" / f"causal_c3_{model}_grid_confirm_h4.jsonl"))}
    pairs = [(r, mps[(r["id"], r["direction"], r["coef"])]) for r in got]
    d_clean = [abs(a["m_clean"] - b["m_clean"]) for a, b in pairs]
    d = [a["m"] - b["m"] for a, b in pairs]
    firm = [(a, b) for a, b in pairs if abs(b["m"]) >= 0.5]
    agree = sum((a["m"] > 0) == (b["m"] > 0) for a, b in firm) / max(len(firm), 1)
    res = {"model": model, "records": len(pairs), "max_abs_d_m_clean": round(max(d_clean), 4),
           "median_abs_d_m": round(statistics.median(map(abs, d)), 4), "median_d_m": round(statistics.median(d), 4),
           "max_abs_d_m": round(max(map(abs, d)), 4), "sign_agree_firm": round(agree, 4), "n_firm": len(firm)}
    res["pass"] = (res["max_abs_d_m_clean"] <= 0.3 and res["median_abs_d_m"] <= 0.25 and abs(res["median_d_m"]) <= 0.1
                   and agree >= 0.95)
    (RUNS / "modal" / "out" / model / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def install(model):
    b = bundle(model, "c3")
    src = fetch(model, "c3")
    recs = [json.loads(line) for line in open(src)]
    per_row = len(b["coefs"]) * (4 + 2 * b["seeds"])
    ids = list(dict.fromkeys(r["id"] for r in recs))
    if len(recs) != per_row * len(b["rows"]) or ids != [r["id"] for r in b["rows"]]:
        raise ValueError(f"{model}: {len(recs)} records over {len(ids)} rows; expected {per_row * len(b['rows'])}")
    dest = RUNS / model / "grid" / f"{b['name']}.jsonl"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_bytes(src.read_bytes())
    summ = json.loads((src.parent / f"{b['name']}.summary.json").read_text())
    (RUNS / "logs" / f"c3_{model}.log").write_text(json.dumps(summ["summary"]) + "\n" + json.dumps(summ["platform"]) + "\n")
    with open(RUNS / "queue" / "queue_e.done", "a") as f:
        f.write(f"c3_{model} exit=0 {datetime.now():%a %b %d %H:%M:%S %Y} modal\n")
    print(f"installed {dest}")


def status():
    now = time.time()
    for k, v in sorted(progress.items()):
        age = round((now - v.get("updated", now)) / 60, 1)
        stale = " STALE?" if v.get("state") == "running" and age > 10 else ""
        print(f"{k:28s} {v.get('state', ''):13s} rows {v.get('rows_done', '-')}/{v.get('rows_total', '-')}  "
              f"{v.get('billed_seconds', 0) / 60:6.1f} min  ${v.get('usd', 0):6.3f}  "
              f"{v.get('sec_per_forward', '')} s/fwd  updated {age} min ago{stale}")
    print(f"spent about ${spent():.2f} of the ${CAP_USD:.0f} cap")


def launch(fn, model, mode, max_minutes, run_id):
    if fn == "ladder" and spent() + max_minutes * 60 * GPU_USD_S > CAP_USD:
        raise SystemExit(f"refused: ${spent():.2f} spent, this call could add ${max_minutes * 60 * GPU_USD_S:.2f}")
    f = modal.Function.from_name(APP, fn)
    call = f.spawn(model) if fn != "ladder" else f.spawn(model, mode, max_minutes * 60, run_id)
    rec = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "fn": fn, "model": model, "mode": mode, "max_minutes": max_minutes,
           "run_id": run_id, "call": call.object_id}
    (RUNS / "modal").mkdir(exist_ok=True)
    with open(RUNS / "modal" / "calls.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("fn", choices=["download", "download_gated", "ladder"])
    p.add_argument("model")
    p.add_argument("--mode", default="c3", choices=["c3", "parity"])
    p.add_argument("--max-minutes", type=float, default=15)
    p.add_argument("--run-id", default="")
    sub.add_parser("status")
    for name in ("parity", "install"):
        sub.add_parser(name).add_argument("model")
    a = ap.parse_args()
    if a.cmd == "launch":
        launch(a.fn, a.model, a.mode, a.max_minutes, a.run_id or f"{datetime.now():%Y%m%d-%H%M%S}")
    elif a.cmd == "status":
        status()
    elif a.cmd == "parity":
        parity(a.model)
    else:
        install(a.model)


if __name__ == "__main__":
    cli()
