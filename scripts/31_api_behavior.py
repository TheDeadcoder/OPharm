import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import opharm
import httpx
import yaml
from google import genai
from google.genai import errors

from opharm.bench.tools import TOOLS
from opharm.paths import BENCH, CONFIGS, ROOT, RUNS
from opharm.run.api import gemini_call, gemini_client, serialize

sys.path.insert(0, str(ROOT / "scripts"))
RETRY = (429, 500, 502, 503, 504)
ATTEMPTS = 6


def answer_of(text):
    word = text.strip().split()[0].strip(".,:!*").lower() if text.strip() else ""
    return word if word in ("yes", "no") else "other"


def cost(usage, spec):
    return (usage["input"] * spec["usd_per_m_input"] + (usage["output"] + usage["thinking"]) * spec["usd_per_m_output"]) / 1e6


def select(args):
    sets, policies, cells, questions = (set(v.split(",")) for v in (args.sets, args.policies, args.cells, args.questions))
    rows = [r for r in map(json.loads, open(args.instances)) if r["set"] in sets and (args.split == "all" or r["split"] == args.split)
            and ("policy" not in r or r["policy"] in policies) and ("target" not in r or r["target"] + r["env"] in cells)
            and (r["set"] != "judge" or r["question"] in questions)]
    if args.skeletons:
        keep = importlib.import_module("09_run_behavior").stratified(rows, args.skeletons)
        rows = [r for r in rows if r["skeleton"] in keep]
    return rows[: args.limit] if args.limit else rows


def call(client, spec, row, tools, level):
    for k in range(ATTEMPTS):
        try:
            return gemini_call(client, spec["model"], row, tools, level)
        except (errors.APIError, httpx.TransportError) as e:
            if (isinstance(e, errors.APIError) and e.code not in RETRY) or k == ATTEMPTS - 1:
                raise
            time.sleep(min(60.0, 2 ** k + random.random()))


def sanity(out, rows, model):
    from opharm.bench.labels import label
    meta, counts = {r["id"]: r for r in rows}, Counter()
    for rec in map(json.loads, open(out / "results.jsonl")):
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
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    spec = yaml.safe_load((CONFIGS / "api_models.yaml").read_text())[args.model]
    rows = select(args)
    out = RUNS / args.model / args.tag
    out.mkdir(parents=True, exist_ok=True)
    order = [r["id"] for r in rows]
    if (out / "order.json").exists() and json.loads((out / "order.json").read_text()) != order:
        raise ValueError("run directory holds a different instance order; use a new tag")
    (out / "order.json").write_text(json.dumps(order))
    kept = out / ("samples.jsonl" if args.samples > 1 else "results.jsonl")
    previous = list(map(json.loads, open(kept))) if kept.exists() else []
    done = {(r["id"], r.get("sample", 0)) for r in previous}
    spent, totals, versions = sum(cost(r["usage"], spec) for r in previous), Counter(), Counter()
    for r in previous:
        totals.update(r["usage"])
        versions[r["model_version"]] += 1
    todo = iter([(r, s) for r in rows for s in range(args.samples) if (r["id"], s) not in done])
    old = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    start, n_err, stopped, pending = old.get("utc_start", datetime.now(timezone.utc).isoformat(timespec="seconds")), 0, spent >= args.max_usd, {}
    client = gemini_client(spec["location"])
    names = ("results", "raw", "errors") + (("samples",) if args.samples > 1 else ())
    files = {k: open(out / f"{k}.jsonl", "a") for k in names}
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            while not stopped and len(pending) < args.workers and (nxt := next(todo, None)) is not None:
                r, s = nxt
                tools = None if args.no_tools or r["set"] == "flat" else TOOLS
                pending[pool.submit(call, client, spec, r, tools, args.thinking_level)] = nxt
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                r, s = pending.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    files["errors"].write(json.dumps({"id": r["id"], "sample": s, "error": f"{type(e).__name__}: {e}"}) + "\n")
                    n_err += 1
                    continue
                rec = {"id": r["id"], "set": r["set"], "text": serialize(res["text"], res["calls"]), "n_calls": len(res["calls"]),
                       "usage": res["usage"], "stop_reason": res["stop_reason"], "model_version": res["model_version"]}
                if r["set"] == "judge":
                    rec.update(answer=answer_of(res["text"]), judge_text=res["text"])
                files["raw"].write(json.dumps({"id": r["id"], "sample": s, "response": res["raw"]}) + "\n")
                if args.samples > 1:
                    files["samples"].write(json.dumps({**rec, "sample": s}) + "\n")
                if s == 0:
                    files["results"].write(json.dumps(rec) + "\n")
                spent += cost(res["usage"], spec)
                totals.update(res["usage"])
                versions[res["model_version"]] += 1
                stopped = stopped or spent >= args.max_usd
            for f in files.values():
                f.flush()
    for f in files.values():
        f.close()
    n_done = sum(1 for _ in open(kept)) if kept.exists() else 0
    manifest = {"model": args.model, "provider": spec["provider"], "provider_model": spec["model"], "location": spec["location"],
                "model_versions": dict(versions), "sdk": {"name": "google-genai", "version": genai.__version__}, "args": vars(args),
                "instances_sha256": hashlib.sha256(open(args.instances, "rb").read()).hexdigest(),
                "utc_start": start, "utc_end": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_selected": len(rows), "n_calls_planned": len(rows) * args.samples, "n_calls_done": n_done, "n_errors_this_session": n_err,
                "tokens": {k: totals[k] for k in ("input", "output", "thinking")}, "cost_usd": round(spent, 4),
                "prices_usd_per_m": {"input": spec["usd_per_m_input"], "output_and_thinking": spec["usd_per_m_output"]},
                "stopped_at_budget": n_done < len(rows) * args.samples and spent >= args.max_usd}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{n_done} of {len(rows) * args.samples} calls done, {n_err} errors this session, cost ${spent:.4f}")
    if manifest["stopped_at_budget"]:
        print(f"stopped at the budget of ${args.max_usd}; rerun with a higher --max-usd to continue")
    if (out / "results.jsonl").exists():
        sanity(out, rows, args.model)


if __name__ == "__main__":
    main()
