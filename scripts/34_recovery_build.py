import argparse
import hashlib
import json

import opharm

from opharm.paths import BENCH, RESULTS
from opharm.recovery.fixtures import DRILL, FIXTURES, HEALTH, RULE
from opharm.recovery.probe import EVIDENCE, instances


def review(rows):
    text = {r["id"]: "".join(t for _, t in r["user"]) for r in rows}
    out = ["# Recovery probe fixtures", "", f"Rule: {RULE}", ""]
    for r in rows:
        if not r["id"].endswith(".main.R2R"):
            continue
        last, health = text[r["id"].replace(".main.R2R", ".main.L2R")], text[r["id"].replace(".main.R2R", ".main.R1R")]
        out += [f"## {r['skeleton']} ({r['cls']}, {r['tool']})", "", "Redundant world, restore drill:", "", "```", text[r["id"]], "```", "", "Last-path world replaces:", ""]
        out += [f"- `{a}`\n  with `{b}`" for a, b in zip(text[r["id"]].split("\n"), last.split("\n")) if a != b]
        out += ["", "Health preview (both worlds):", "", "- `" + next(x for x in health.split("\n") if x.startswith("Preflight")) + "`", ""]
    return "\n".join(out)


def render_check(rows, models):
    from opharm.bench.tools import TOOLS
    from opharm.chat import load_tokenizer, render
    for model in models:
        tok = load_tokenizer(model)
        n = [len(render(tok, r["system"], r["user"], TOOLS, thinking=t).ids) for r in rows for t in (False, True)]
        print(model, "renders", len(n), "tokens min", min(n), "max", max(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--render", default="")
    args = ap.parse_args()
    rows, truths = instances()
    body = "".join(json.dumps(r) + "\n" for r in rows)
    BENCH.mkdir(exist_ok=True)
    (BENCH / "recovery.jsonl").write_text(body)
    out = {"designed_after_main_results": True, "rule": RULE, "evidence": list(EVIDENCE), "report_templates": {"health": HEALTH, "drill": DRILL},
           "n_instances": len(rows), "n_templates": len(FIXTURES), "sha256_instances": hashlib.sha256(body.encode()).hexdigest(),
           "fixtures": {fx.id: {"mechanism": fx.mechanism, "tool": fx.tool, "command": fx.command, "defect": list(fx.defect),
                                "worlds": {k.split(".", 1)[1]: v for k, v in truths.items() if k.split(".")[0] == fx.id}} for fx in FIXTURES}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "recovery_fixtures.json").write_text(json.dumps(out, indent=1))
    print("wrote", BENCH / "recovery.jsonl", len(rows), "instances", out["sha256_instances"])
    for fx in FIXTURES:
        t = out["fixtures"][fx.id]["worlds"]
        print(f"{fx.id:18s}", " ".join(f"{k}: {v['restored_with_command']}/{v['total']}" for k, v in sorted(t.items())))
    if args.review:
        (BENCH / "recovery_review.md").write_text(review(rows))
        print("wrote", BENCH / "recovery_review.md")
    if args.render:
        render_check(rows, args.render.split(","))


if __name__ == "__main__":
    main()
