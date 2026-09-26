import argparse
import json

import opharm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opharm.paths import RESULTS

OUT = RESULTS / "figures"
COLORS = {"prod": "#c0392b", "stage": "#2e86c1", "ref": "#7d3c98", "blast": "#d35400", "perp": "#e59866",
          "inst": "#117a65", "rand": "#7f8c8d", "span": "#2e86c1", "rest": "#95a5a6"}


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def err(v):
    return [[v[0] - v[1][0]], [v[1][1] - v[0]]]


def behavior(grid, name):
    cells = grid["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.1]}, layout="constrained")
    for ax, target, title in ((axes[0], "D", "Destructive target"), (axes[1], "B", "Benign target")):
        groups = [("C", "A"), ("C", "N"), ("N", "A"), ("N", "N")]
        x = np.arange(len(groups))
        for k, (env, color, label) in enumerate((("P", COLORS["prod"], "production"), ("S", COLORS["stage"], "staging"))):
            vals = [cells[target + env + rb + pol].get("EXEC_MATCH", 0) for pol, rb in groups]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=label)
        ax.set_xticks(x, ["confirm rule\nrollback", "confirm rule\nno rollback", "neutral rule\nrollback", "neutral rule\nno rollback"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Executed as requested")
    axes[0].legend(fontsize=8, loc="lower left")
    g = grid["gate_g2"]
    effects = [("Confirmation policy", g["m_policy_effect_neutral_minus_confirm"]),
               ("Rollback", grid["m_rollback_effect_available_minus_none"]),
               ("Target", grid["m_target_effect_benign_minus_destructive"]),
               ("Blast radius (neutral rule)", g["m_blast_effect_stage_minus_prod"]["N"]),
               ("Blast radius (confirm rule)", g["m_blast_effect_stage_minus_prod"]["C"])]
    ax = axes[2]
    for i, (label, v) in enumerate(effects):
        ax.barh(i, v[0], xerr=err(v), color=COLORS["ref"] if i == 0 else COLORS["blast"], capsize=3)
    ax.set_yticks(range(len(effects)), [e[0] for e in effects], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Effect on m(x), logits")
    ax.set_title("Factor effects on the decision to act", fontsize=10)
    save(fig, f"fig1_behavior_{name}")


def representation(rep, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
    types = rep["point_block_type"]
    ax = axes[0]
    for key, label, color in (("harm_t_inst", "r_harm at t_inst", COLORS["ref"]), ("ref_t_post", "r_ref at t_post", COLORS["blast"])):
        d = rep["seen"][key]["delta_by_layer"]
        ax.plot(range(len(d)), d, color=color, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Production shift (content-harm gaps)")
    ax.set_title("Seen: production shift on the content-harm directions", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for p, color in (("t_inst", COLORS["inst"]), ("t_post", COLORS["blast"])):
        scan = rep["represented"]["scan"][p]
        ax.plot([d["layer"] for d in scan], [d["test_auroc"] for d in scan], color=color, label=f"probe at {p}")
    ax.axhline(max(rep["represented"]["ngram_env"].values()), color=COLORS["rand"], ls="--", label="n-gram baseline")
    for i, t in enumerate(types):
        if t == "attention":
            ax.axvline(i, color="grey", lw=0.4, alpha=0.5)
    for key, marker in (("registered", "o"), ("dev_best", "s")):
        pt = rep["represented"]["points"][key]
        ax.plot(pt["layer"], pt["test_auroc"], marker, color="black", ms=6, label=f"{key.replace('_', ' ')} point")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Layer (grey lines: attention layers)")
    ax.set_ylabel("AUROC, unseen classes and forms")
    ax.set_title("Represented: blast-radius probe", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, f"fig2_representation_{name}")


def causal(cz, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    ax = axes[0]
    series = [("c1", "", "environment span (C1)", COLORS["span"]), ("c7", "", "all other positions (C7)", COLORS["rest"]),
              ("c6", "|r_blast", "t_inst production coordinate (C6)", COLORS["inst"]),
              ("c6_post", "|r_blast_post", "t_post production coordinate (C6)", COLORS["blast"])]
    for key, suffix, label, color in series:
        if key not in cz:
            continue
        rows = sorted((int(k.split("|")[0]), v["m_fraction"]) for k, v in cz[key].items()
                      if k.split("|")[1] == "P<-S" and k.endswith(suffix) and v["m_fraction"])
        x, v = [r[0] for r in rows], np.array([r[1] for r in rows])
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=color, marker="o", ms=3, capsize=2, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(1, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("Share of the production effect on m(x)")
    ax.set_title("Where the production effect travels", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1]
    table = cz["c3"]["mean_m_shift"]
    coefs = sorted({float(k.split("|")[1]) for k in table})
    for kind, label, color in (("r_ref", "r_ref", COLORS["ref"]), ("r_blast", "r_blast (t_post)", COLORS["blast"]),
                               ("r_blast_perp", "r_blast without r_ref part", COLORS["perp"]),
                               ("r_blast_inst", "r_blast (t_inst)", COLORS["inst"]), ("rand_ref", "random", COLORS["rand"])):
        v = np.array([table[f"{kind}|{c}|DP"] for c in coefs])
        ax.errorbar(coefs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=color, marker="o", ms=3, capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_xticks(coefs, [str(c) for c in coefs])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Steering coefficient (units of the r_ref norm)")
    ax.set_ylabel("Change in m(x), destructive production")
    ax.set_title("Steering at layer 18, matched norm", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, f"fig3_causal_{name}")


def vocabulary(grid, name):
    v = grid["vocabulary"]
    rows = [("Plain consequence note\n(destructive target)", v["ladder_neutral_minus_none"], COLORS["stage"]),
            ("Harm wording beyond\nthe plain note", v["ladder_harm_minus_neutral"], COLORS["ref"]),
            ("Harm wording on a\nbenign target", v["cue_harm_minus_none"], COLORS["ref"]),
            ("Slang on a\nbenign target", v["cue_slang_minus_none"], COLORS["rand"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout="constrained")
    for i, (label, e, color) in enumerate(rows):
        ax.barh(i, e[0], xerr=err(e), color=color, capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Change in m(x), logits")
    ax.set_title("Vocabulary against consequence", fontsize=10)
    save(fig, f"fig4_vocabulary_{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    sfx = "_confirm" if args.confirm else ""
    name = f"{args.model}{sfx}"
    grid, rep = load(f"{args.tag}_{args.model}{sfx}"), load(f"representation_{args.model}_{args.tag}{sfx}")
    cz = load(f"causal_{args.model}_{args.tag}{sfx}")
    made = []
    if grid:
        behavior(grid, name)
        made.append("fig1")
        if grid.get("vocabulary"):
            vocabulary(grid, name)
            made.append("fig4")
    if rep:
        representation(rep, name)
        made.append("fig2")
    if cz and "c3" in cz:
        causal(cz, name)
        made.append("fig3")
    print(name, made)


if __name__ == "__main__":
    main()
