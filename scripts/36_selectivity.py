import argparse
import importlib
import json
import sys

import opharm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from opharm.bench.evaluate import ENV_WORDS
from opharm.paths import BENCH, RESULTS, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
rv = importlib.import_module("16_rule_variants")

LOCAL = ("qwen35_4b", "qwen35_9b", "qwen3_4b_2507", "gemma4_e4b", "llama31_8b")
NAMES = {"qwen35_4b": "Qwen3.5-4B", "qwen35_9b": "Qwen3.5-9B", "qwen3_4b_2507": "Qwen3-4B-2507", "gemma4_e4b": "Gemma-4-E4B",
         "llama31_8b": "Llama-3.1-8B", "gemini38_flash": "Gemini 3.8 Flash", "gemini31_pro": "Gemini 3.1 Pro"}
MODE_COLOR = {"direct": "#2a78d6", "narration": "#eb6834", "thinking": "#1baf7a"}
INK, QUIET, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MARKER = {"registered": "o", "rule_variant": "^", "trigger": "s", "recovery": "D", "self_check": "*", "regex_guard": "X"}
FAMILY = {"registered": "registered rule", "rule_variant": "other rule wordings", "trigger": "harmless trigger rule",
          "recovery": "recovery rule (probe)", "self_check": "self-check wrapper (simulated)", "regex_guard": "regex guard on the environment word"}
CELLS, FREE = ("DP", "DS", "BP", "BS"), ("DS", "BP", "BS")


def read(name):
    for stem in (name + "_final", name):
        path = RESULTS / f"{stem}.json"
        if path.exists():
            return json.loads(path.read_text()), stem
    return {}, None


def est(entry):
    return None if entry is None else entry["estimate"]


def row(model, family, condition, mode, split, need, over, source, paired=None, **extra):
    sel = paired["estimate"] if paired else None if need is None or over is None else round(need - over, 4)
    return {"model": model, "family": family, "condition": condition, "mode": mode, "split": split, "required_ask": need, "over_ask": over,
            "selectivity": sel, "selectivity_ci_template": (paired or {}).get("ci_template"), "source": source, **extra}


def collect(models):
    rows = []
    for m in models:
        modes, src = read(f"modes_{m}")
        for mode, s in modes.get("splits", {}).get("heldout", {}).get("modes", {}).items():
            rows.append(row(m, "registered", "registered", mode, "heldout", est(s["cells"]["DPC"]["ask"]), est(s["over_ask"]), src, s["ask_selectivity"],
                            required_ci_template=(s["cells"]["DPC"]["ask"] or {}).get("ci_template"), over_ci_template=(s["over_ask"] or {}).get("ci_template"),
                            required_ask_covered=est(s["covered_DPC"]["ask"]), paired_correct=est(s["paired_correct_abstention"])))
        rules, src = read(f"rules_{m}_v2")
        for name, c in rules.get("conditions", {}).items():
            need = [x for x in CELLS if rv.gold(name, x) == "abstain"]
            free = [x for x in CELLS if rv.gold(name, x) == "act"]
            if name in ("registered", "trigger_present", "trigger_absent") or not need or not free:
                continue
            mean = lambda cells: round(float(np.mean([c[f"{x}_asks"][0] for x in cells])), 4)
            rows.append(row(m, "rule_variant", name, "direct", "dev", mean(need), mean(free), src))
        trig, src = read(f"trigger_{m}")
        for mode, s in trig.get("runs", {}).items():
            a, b = s["by_variant"]["trigger_present.all"]["ask"], s["by_variant"]["trigger_absent.all"]["ask"]
            rows.append(row(m, "trigger", "trigger", mode, "dev", est(a), est(b), src, s["paired"]["all"]["ask_present_minus_absent"],
                            cells=s.get("cells"), required_ci_template=(a or {}).get("ci_template"),
                            over_ci_template=(b or {}).get("ci_template"), paired_correct=est(s["paired"]["all"]["paired_correct"])))
        rec, src = read(f"recovery_{m}")
        for mode, s in rec.get("runs", {}).items():
            for e in ("state", "health", "drill"):
                a, b = s["cells"][f"R.{e}.L"]["ask"], s["cells"][f"R.{e}.R"]["ask"]
                rows.append(row(m, "recovery", e, mode, "probe", est(a), est(b), src, s["selectivity"][f"R.{e}"], required_ci_template=(a or {}).get("ci_template"),
                                over_ci_template=(b or {}).get("ci_template"), paired_correct=est(s["paired_correct"][f"R.{e}"])))
        sec, src = read(f"secondary_{m}")
        check = sec.get("splits", {}).get("heldout", {}).get("self_check_from_tool_free_q1")
        if check and all(check[x + "C"]["self_check_ask"] for x in CELLS):
            rows.append(row(m, "self_check", "ask when its own q1 answer is yes", "simulated", "heldout", check["DPC"]["self_check_ask"][0],
                            round(float(np.mean([check[x + "C"]["self_check_ask"][0] for x in FREE])), 4), src))
    if (BENCH / "instances.jsonl").exists():
        flag = {x: [] for x in CELLS}
        for r in map(json.loads, open(BENCH / "instances.jsonl")):
            if r["set"] == "main" and r["split"] == "heldout" and r["policy"] == "C":
                flag[r["target"] + r["env"]].append(bool(ENV_WORDS["P"].search("".join(t for _, t in r["user"]))))
        rows.append(row("none", "regex_guard", "ask when the ticket names production", "none", "heldout", round(float(np.mean(flag["DP"])), 4),
                        round(float(np.mean([v for x in FREE for v in flag[x]])), 4), "benchmark/instances.jsonl"))
    return rows


def figure(rows, models):
    shown = [m for m in models if any(r["model"] == m for r in rows)]
    cols = min(3, len(shown))
    lines = -(-len(shown) // cols)
    fig, axes = plt.subplots(lines, cols, figsize=(3.1 * cols, 3.25 * lines + 0.9), sharex=True, sharey=True, squeeze=False, layout="constrained")
    guard = next((r for r in rows if r["family"] == "regex_guard"), None)
    for ax, m in zip(axes.ravel(), shown):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
        own = [r for r in rows if r["model"] == m and r["required_ask"] is not None and r["over_ask"] is not None]
        path = [r for mode in ("direct", "narration", "thinking") for r in own if r["family"] == "registered" and r["mode"] == mode]
        ax.plot([r["over_ask"] for r in path], [r["required_ask"] for r in path], color=QUIET, lw=0.9, zorder=2)
        for r in sorted(own, key=lambda r: r["family"] != "rule_variant"):
            small, drill = r["family"] == "rule_variant", r["family"] == "recovery" and r["condition"] == "drill"
            ax.plot(r["over_ask"], r["required_ask"], MARKER[r["family"]], color=MODE_COLOR.get(r["mode"], QUIET), ms=5 if small else 8 if r["family"] != "self_check" else 11,
                    mec=INK if drill else "white", mew=1.3 if drill else 1.0, alpha=0.75 if small else 1.0, zorder=5 if drill else 3 if small else 4, clip_on=False)
        if guard:
            ax.plot(guard["over_ask"], guard["required_ask"], MARKER["regex_guard"], color=INK, ms=7, mec="white", mew=0.8, zorder=4, clip_on=False)
        ax.set_title(NAMES.get(m, m), fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.tick_params(labelsize=8, colors=QUIET, length=0)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ax.spines.values():
            side.set_visible(False)
    first = axes[0, 0]
    first.text(0.02, 1.0, "ideal", fontsize=7.5, color=QUIET, va="top")
    first.text(0.60, 0.50, "no selectivity", fontsize=7.5, color=QUIET, rotation=45, rotation_mode="anchor", ha="center", va="top")
    for ax in axes.ravel()[len(shown):]:
        ax.set_visible(False)
    for column in axes.T:
        last = [ax for ax in column if ax.get_visible()][-1]
        last.tick_params(labelbottom=True)
        last.set_xlabel("asks where the rule does not require it", fontsize=8.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("asks where the rule requires it", fontsize=8.5, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=c, mec="white", ms=8, label=f"{k} mode") for k, c in MODE_COLOR.items()]
    handles.append(Line2D([], [], color=QUIET, lw=0.9, label="registered rule across modes"))
    handles += [Line2D([], [], marker=MARKER[f], ls="", color=INK if f == "regex_guard" else QUIET, mec="white", ms=11 if f == "self_check" else 8,
                       label=FAMILY[f]) for f in MARKER if any(r["family"] == f for r in rows)]
    if any(r["family"] == "recovery" for r in rows):
        at = next(i for i, h in enumerate(handles) if h.get_label() == FAMILY["recovery"]) + 1
        handles.insert(at, Line2D([], [], marker="D", ls="", color=QUIET, mec=INK, mew=1.3, ms=8, label="recovery rule with the restore drill"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False, labelcolor=INK)
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"selectivity.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(LOCAL) + ",gemini38_flash")
    args = ap.parse_args()
    models = args.models.split(",")
    rows = collect(models)
    out = {"sources": "a file with the suffix _final is used when it exists; each row names its source",
           "definitions": {"required_ask": "share of prompts answered with a confirmation request where the rule in force requires one",
                           "over_ask": "the same share where the rule in force does not require one",
                           "registered": "DP against DS, BP and BS under the registered rule, held-out", "rule_variant": "cells per rule wording as in 16_rule_variants",
                           "trigger": "trigger present against trigger absent, all cells", "recovery": "last restore path against redundant path, by evidence level",
                           "self_check": "simulated policy that asks whenever the model's tool-free answer to q1 is yes",
                           "regex_guard": "asks whenever the ticket text matches the production word pattern of the evaluator"},
           "rows": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selectivity.json").write_text(json.dumps(out, indent=1))
    figure(rows, models)
    for r in rows:
        print(f"{r['model']:15s} {r['family']:13s} {r['condition'][:22]:22s} {r['mode']:10s} required {r['required_ask']} over {r['over_ask']} selectivity {r['selectivity']}")
    print("wrote", RESULTS / "selectivity.json", "and", RESULTS / "figures" / "selectivity.png")


if __name__ == "__main__":
    main()
