import argparse
import atexit
import hashlib
import importlib
import json
import os
import random
import re
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import opharm
import google.auth.exceptions as gauth
import httpx
import yaml
from google import genai
from google.genai import errors

from opharm.bench.tools import TOOLS
from opharm.paths import BENCH, CONFIGS, ROOT, RUNS
from opharm.run.api import gemini_call, gemini_client, serialize

sys.path.insert(0, str(ROOT / "scripts"))
TRANSIENT = {408, 429, 500, 502, 503, 504}
BACKOFF_START, BACKOFF_CAP = 10.0, 300.0
EXIT_DONE, EXIT_FAILED, EXIT_BUDGET, EXIT_OUTAGE, EXIT_PERMANENT, EXIT_STOPPED = 0, 1, 2, 3, 4, 130


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def answer_of(text):
    word = text.strip().split()[0].strip(".,:!*").lower() if text.strip() else ""
    return word if word in ("yes", "no") else "other"


def cost(usage, spec):
    return (usage["input"] * spec["usd_per_m_input"] + (usage["output"] + usage["thinking"]) * spec["usd_per_m_output"]) / 1e6


def transient(e):
    if isinstance(e, errors.APIError):
        return e.code in TRANSIENT
    return isinstance(e, (httpx.TransportError, gauth.TransportError, ConnectionError, TimeoutError)) or \
        (type(e).__module__.startswith("httpx") and any(w in type(e).__name__ for w in ("Timeout", "Connect", "Network", "Protocol", "Read", "Write")))


def server_delay(e):
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', json.dumps(getattr(e, "details", None) or {}))
    return float(m.group(1)) if m else 0.0


def read_jsonl(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    keep = data[: data.rfind(b"\n") + 1]
    if len(keep) != len(data):
        path.write_bytes(keep)
        print(f"dropped a torn last line in {path.name}")
    return [json.loads(line) for line in keep.splitlines() if line.strip()]


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, obj):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


class Gate:
    def __init__(self):
        self.until, self.lock = 0.0, threading.Lock()

    def hold(self, seconds):
        with self.lock:
            self.until = max(self.until, time.time() + seconds)

    def wait(self, stop):
        while not stop.is_set() and (delay := self.until - time.time()) > 0:
            time.sleep(min(delay, 2.0))


def take_lock(out):
    lock = out / "run.lock"
    if lock.exists():
        pid = int(lock.read_text().strip() or 0)
        try:
            os.kill(pid, 0)
            alive = pid > 0
        except OSError:
            alive = False
        if alive:
            raise SystemExit(f"process {pid} is already running {out}")
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.exists() and lock.read_text().strip() == str(os.getpid()) and lock.unlink())


def select(args):
    sets, policies, cells, questions = (set(v.split(",")) for v in (args.sets, args.policies, args.cells, args.questions))
    rows = [r for r in map(json.loads, open(args.instances)) if r["set"] in sets and (args.split == "all" or r["split"] == args.split)
            and ("policy" not in r or r["policy"] in policies) and ("target" not in r or r["target"] + r["env"] in cells)
            and (r["set"] != "judge" or r["question"] in questions)]
    if args.skeletons:
        keep = importlib.import_module("09_run_behavior").stratified(rows, args.skeletons)
        rows = [r for r in rows if r["skeleton"] in keep]
    return rows[: args.limit] if args.limit else rows


def call(client, spec, row, tools, level, gate, window, stop):
    start, k = time.time(), 0
    while True:
        gate.wait(stop)
        try:
            return gemini_call(client, spec["model"], row, tools, level), k + 1
        except Exception as e:
            e.attempts = k + 1
            if not transient(e) or stop.is_set() or time.time() - start > window:
                raise
            delay = max(min(BACKOFF_CAP, BACKOFF_START * 2.0 ** min(k, 6)) * (0.75 + 0.5 * random.random()), server_delay(e))
            if not isinstance(e, errors.APIError) or e.code == 429:
                gate.hold(delay)
            print(f"{now()} retry {row['id']} attempt {k + 2} in {delay:.0f}s after {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(delay)
            k += 1


def sanity(out, rows, model):
    from opharm.bench.labels import label
    meta, counts = {r["id"]: r for r in rows}, Counter()
    for rec in read_jsonl(out / "results.jsonl"):
        r = meta.get(rec["id"])
        if r is not None and r["set"] in ("main", "narr"):
            counts[(r["set"], r["policy"], r["target"] + r["env"], label(rec["text"], r, model, None)["label"])] += 1
    for k, n in sorted(counts.items()):
        print(*k, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--instances", default=str(BENCH / "instances.jsonl"))
    ap.add_argument("--sets", default="main")
    ap.add_argument("--split", default="heldout", choices=["dev", "heldout", "all"])
    ap.add_argument("--policies", default="C,N")
    ap.add_argument("--cells", default="DP,DS,BP,BS")
    ap.add_argument("--questions", default="q1,q2,q3")
    ap.add_argument("--skeletons", type=int, default=0)
    ap.add_argument("--thinking-level", default="low")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retry-window", type=float, default=1800.0)
    ap.add_argument("--timeout", type=float, default=0.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=6)
    args = ap.parse_args()
    if not os.path.isabs(args.instances) and not os.path.exists(args.instances):
        args.instances = str(BENCH / args.instances)
    spec = yaml.safe_load((CONFIGS / "api_models.yaml").read_text())[args.model]
    rows = select(args)
    out = RUNS / args.model / args.tag
    out.mkdir(parents=True, exist_ok=True)
    take_lock(out)
    order = [r["id"] for r in rows]
    if (out / "order.json").exists() and json.loads((out / "order.json").read_text()) != order:
        raise ValueError("run directory holds a different instance order; use a new tag")
    (out / "order.json").write_text(json.dumps(order))
    kept = out / ("samples.jsonl" if args.samples > 1 else "results.jsonl")
    previous = read_jsonl(kept)
    if args.samples > 1:
        have = {r["id"] for r in read_jsonl(out / "results.jsonl")}
        fix = Log(out / "results.jsonl")
        for r in previous:
            if r.get("sample") == 0 and r["id"] not in have:
                fix.write({k: v for k, v in r.items() if k != "sample"})
        fix.close()
    read_jsonl(out / "raw.jsonl")
    read_jsonl(out / "errors.jsonl")
    done = {(r["id"], r.get("sample", 0)) for r in previous}
    spent, totals, versions = sum(cost(r["usage"], spec) for r in previous), Counter(), Counter()
    for r in previous:
        totals.update(r["usage"])
        versions[r["model_version"]] += 1
    todo = iter([(r, s) for r in rows for s in range(args.samples) if (r["id"], s) not in done])
    old = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    inst_sha = hashlib.sha256(open(args.instances, "rb").read()).hexdigest()
    code_sha = hashlib.sha256(b"".join(open(f, "rb").read() for f in (__file__, ROOT / "src" / "opharm" / "run" / "api.py"))).hexdigest()
    start, session = old.get("utc_start", now()), now()
    stop, gate, pending = threading.Event(), Gate(), {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (print("stop requested; finishing the calls in flight"), stop.set()))
    counts = Counter(done=len(done))
    status = {"budget": spent >= args.max_usd, "outage": False, "fatal": None}

    def manifest(final=False):
        n_done = counts["done"]
        body = {"model": args.model, "provider": spec["provider"], "provider_model": spec["model"], "location": spec["location"],
                "model_versions": dict(versions), "sdk": {"name": "google-genai", "version": genai.__version__}, "args": vars(args),
                "instances_sha256": inst_sha, "code_sha256": code_sha, "utc_start": start,
                "utc_session_start": session, "utc_end": now() if final else None, "running": not final,
                "n_selected": len(rows), "n_calls_planned": len(rows) * args.samples, "n_calls_done": n_done,
                "errors_this_session": {"transient": counts["transient"], "permanent": counts["permanent"]},
                "tokens": {k: totals[k] for k in ("input", "output", "thinking")}, "cost_usd": round(spent, 4),
                "prices_usd_per_m": {"input": spec["usd_per_m_input"], "output_and_thinking": spec["usd_per_m_output"]},
                "stopped_at_budget": status["budget"] and n_done < len(rows) * args.samples, "stopped_after_outage": status["outage"]}
        tmp = out / "manifest.json.tmp"
        tmp.write_text(json.dumps(body, indent=1))
        os.replace(tmp, out / "manifest.json")

    timeout = args.timeout or (1800.0 if args.thinking_level == "high" else 600.0)
    window = max(args.retry_window, 2 * timeout)
    client = gemini_client(spec["location"], timeout)
    logs = {k: Log(out / f"{k}.jsonl") for k in ("results", "raw", "errors") + (("samples",) if args.samples > 1 else ())}
    consecutive, last = 0, time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            while not (stop.is_set() or any(status.values())) and len(pending) < args.workers and (nxt := next(todo, None)) is not None:
                r, s = nxt
                tools = None if args.no_tools or r["set"] == "flat" else TOOLS
                pending[pool.submit(call, client, spec, r, tools, args.thinking_level, gate, window, stop)] = nxt
            if not pending:
                break
            finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for fut in finished:
                r, s = pending.pop(fut)
                try:
                    res, attempts = fut.result()
                except Exception as e:
                    kind = "transient" if transient(e) else "permanent"
                    counts[kind] += 1
                    logs["errors"].write({"id": r["id"], "sample": s, "kind": kind, "attempts": getattr(e, "attempts", 1),
                                          "error": f"{type(e).__name__}: {e}"[:2000], "utc": now()})
                    if isinstance(e, gauth.RefreshError):
                        status["fatal"] = "the credentials no longer refresh; run gcloud auth application-default login again"
                    consecutive = consecutive + 1 if kind == "transient" else consecutive
                    status["outage"] = status["outage"] or consecutive >= args.max_consecutive_failures
                    continue
                consecutive = 0
                rec = {"id": r["id"], "set": r["set"], "text": serialize(res["text"], res["calls"]), "n_calls": len(res["calls"]),
                       "usage": res["usage"], "stop_reason": res["stop_reason"], "model_version": res["model_version"], "attempts": attempts}
                if r["set"] == "judge":
                    rec.update(answer=answer_of(res["text"]), judge_text=res["text"])
                logs["raw"].write({"id": r["id"], "sample": s, "response": res["raw"]})
                if args.samples > 1:
                    logs["samples"].write({**rec, "sample": s})
                if s == 0:
                    logs["results"].write(rec)
                counts["done"] += 1
                spent += cost(res["usage"], spec)
                totals.update(res["usage"])
                versions[res["model_version"]] += 1
                status["budget"] = status["budget"] or spent >= args.max_usd
            if time.time() - last > 60:
                manifest()
                last = time.time()
    for log in logs.values():
        log.close()
    manifest(final=True)
    planned = len(rows) * args.samples
    print(f"{counts['done']} of {planned} calls done, errors this session: {counts['transient']} transient, {counts['permanent']} permanent, "
          f"cost ${spent:.4f} of ${args.max_usd}")
    if (out / "results.jsonl").exists():
        sanity(out, rows, args.model)
    if status["fatal"]:
        print(status["fatal"])
        sys.exit(EXIT_FAILED)
    if counts["done"] >= planned:
        sys.exit(EXIT_DONE)
    if stop.is_set():
        sys.exit(EXIT_STOPPED)
    if status["budget"]:
        print(f"stopped at the budget of ${args.max_usd}; rerun with a higher --max-usd to continue")
        sys.exit(EXIT_BUDGET)
    if status["outage"]:
        print(f"stopped after {args.max_consecutive_failures} consecutive failed calls; rerun later to resume")
        sys.exit(EXIT_OUTAGE)
    print("some calls failed with errors that retrying did not fix; see errors.jsonl, rerun to retry them")
    sys.exit(EXIT_PERMANENT)


if __name__ == "__main__":
    main()
