import argparse
import hashlib
import json
from collections import Counter

import opharm
import yaml

from opharm.bench import fresh
from opharm.paths import BENCH, CONFIGS, RESULTS

TRUTH = "fresh_truth.yaml"
PROVENANCE = ("Written on 2026-10-02 by human authors. "
              "The authors were completely unaware of the benchmarks. No model had seen these templates.")


def shown(mark):
    return f"{mark['D']} (benign twin {mark['B']})" if isinstance(mark, dict) else mark


def check(truth):
    ids = {t.id for t in fresh.TEMPLATES}
    if set(truth["templates"]) != ids or not set(truth["q3_exclude"]) <= ids:
        raise ValueError(f"configs/{TRUTH} must name exactly the {len(ids)} templates")
    for t, marks in truth["templates"].items():
        values = [v for q in ("QD", "QA", "QO", "QP") for v in (marks[q].values() if isinstance(marks[q], dict) else [marks[q]])]
        if not set(values) <= {"Y", "?", "N"} or not isinstance(marks["snapshot_fits"], bool):
            raise ValueError(f"configs/{TRUTH}: {t} has a mark other than Y, ? or N, or snapshot_fits is not true or false")


def review(skeletons, rows, truth):
    text = {r["id"]: "".join(t for _, t in r["user"]) for r in rows}
    first = {s["template"]: s for s in reversed(skeletons)}
    out = ["# Fresh templates", "", PROVENANCE, ""]
    for tpl in fresh.TEMPLATES:
        mark = truth["templates"][tpl.id]
        sk = first[tpl.id]
        out += [f"## {tpl.id}", "", "```", f"class:               {tpl.cls}", f"tool:                {tpl.tool}", f"destination kind:    {tpl.label}",
                f"purpose:             {tpl.purpose}", f"destructive command: {tpl.d[0][0]}", f"benign command:      {tpl.b[0][0]}",
                f"important names:     {', '.join(tpl.names['fresh'])}" + (f" with {', '.join(tpl.primaries)}" if tpl.primaries != ("",) else ""),
                f"disposable names:    {', '.join(tpl.markers)}", f"operation check:     {tpl.verb}",
                "what it breaks:      " + ", ".join(f"{name} {shown(mark[q])}" for name, q in (("data", "QD"), ("access", "QA"), ("service", "QO"), ("monitoring", "QP"))),
                f"                     {mark['why']}", f"rollback that fits:  {mark['undo']}",
                f"                     the ticket's snapshot line {'fits' if mark['snapshot_fits'] else 'does not fit'} this harm"]
        out += ["question q3:         left out, the twins differ in the destination"] if tpl.id in truth["q3_exclude"] else []
        out += ["```", "",
                "Destructive twin, production, rule C:", "", "```", text[f"{sk['id']}.main.DPAC"], "```", "",
                f"Benign twin runs `{sk['twin']['b_cmd']}`.", ""]
    return "\n".join(out)


def render_check(rows, models):
    from opharm.bench.tools import TOOLS
    from opharm.chat import load_tokenizer, render
    main = [r for r in rows if r["set"] == "main"]
    for model in models:
        tok = load_tokenizer(model)
        n = [len(render(tok, r["system"], r["user"], TOOLS, thinking=t).ids) for r in main for t in (False, True)]
        print(model, "renders", len(n), "tokens min", min(n), "max", max(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--render", default="")
    args = ap.parse_args()
    skeletons = fresh.plan()
    rows = fresh.expand(skeletons)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    truth_bytes = (CONFIGS / TRUTH).read_bytes()
    truth = yaml.safe_load(truth_bytes)
    check(truth)
    BENCH.mkdir(exist_ok=True)
    (BENCH / "fresh.jsonl").write_text(body)
    out = {"designed_after_main_results": True, "provenance": PROVENANCE, "n_templates": len(fresh.TEMPLATES), "n_skeletons": len(skeletons),
           "n_instances": len(rows), "by_set": dict(Counter(r["set"] for r in rows)), "sha256_instances": hashlib.sha256(body.encode()).hexdigest(),
           "truth": {"file": f"configs/{TRUTH}", "sha256": hashlib.sha256(truth_bytes).hexdigest(), "status": truth["status"],
                     "confirmed_by": truth["confirmed_by"], "q3_exclude": truth["q3_exclude"]},
           "templates": {t.id: {"cls": t.cls, "tool": t.tool, "label": t.label, "purpose": t.purpose, "destructive": t.d[0][0], "benign": t.b[0][0],
                                "verb": t.verb, "marks": {q: truth["templates"][t.id][q] for q in ("QD", "QA", "QO", "QP", "snapshot_fits")}} for t in fresh.TEMPLATES}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fresh_templates.json").write_text(json.dumps(out, indent=1))
    print("wrote", BENCH / "fresh.jsonl", len(skeletons), "skeletons", len(rows), "instances", out["sha256_instances"])
    print("sets", out["by_set"], "| truth", out["truth"]["sha256"][:12], "confirmed by", repr(truth["confirmed_by"]))
    if not str(truth["confirmed_by"] or "").strip():
        print(f"marks NOT confirmed: read benchmark/fresh_review.md, set confirmed_by in configs/{TRUTH}, then run this script again")
    if args.review:
        (BENCH / "fresh_review.md").write_text(review(skeletons, rows, truth))
        print("wrote", BENCH / "fresh_review.md")
    if args.render:
        render_check(rows, args.render.split(","))


if __name__ == "__main__":
    main()
