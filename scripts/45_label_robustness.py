import importlib
import json
import sys

import opharm

from opharm.paths import RESULTS, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
ar = importlib.import_module("43_assist_report")

ASSIST = ("qwen35_4b", "qwen35_9b", "qwen3_4b_2507", "qwen35_27b", "gemma4_e4b", "llama31_8b")
TRIGGER = ("qwen35_4b", "qwen35_9b", "qwen3_4b_2507", "gemma4_e4b", "llama31_8b", "qwen35_27b", "gemini38_flash")
RULES = ("qwen35_4b", "qwen3_4b_2507", "qwen35_9b", "gemma4_e4b", "llama31_8b")
MODES = ("qwen35_4b", "qwen35_9b", "qwen3_4b_2507", "gemma4_e4b", "llama31_8b", "gemini38_flash")
MEASURES = {"asks": ar.ask, "not_executed": lambda r: r["label"] != "EXEC_MATCH"}


def pick(name):
    final = RESULTS / f"{name}_final.json"
    return json.loads((final if final.exists() else RESULTS / f"{name}.json").read_text())


def est(d):
    return d["estimate"] if d else None


def contrasts(res):
    out = {f"op.{k}.{m}": d for k, v in res["contrasts"].items() for m, d in (v or {}).items() if d}
    out.update({f"cj.{k}.{c}": d["success"] for k, v in res["cj_contrasts"].items() for c, d in (v or {}).items()})
    rep = (res.get("replication_dev") or {}).get("A2u_vs_A0") or {}
    out.update({f"rep.A2u_vs_A0.{m}": d for m, d in rep.items() if d})
    return out


def brief(d):
    return [d["estimate"], d["ci"], d["reading"]] if d else None


def assist(model):
    runs = {}
    for name, fn in MEASURES.items():
        ar.ask = fn
        runs[name] = ar.analyze(model)
    ar.ask = MEASURES["asks"]
    a, b = runs["asks"], runs["not_executed"]
    ca, cb = contrasts(a), contrasts(b)
    changed = {k: {"asks": brief(ca[k]), "not_executed": brief(cb.get(k))} for k in ca if ca[k]["reading"] != (cb.get(k) or {}).get("reading")}
    return {"success": {arm: [est(a["op"][arm]["success"]), est(b["op"][arm]["success"])] for arm in a["op"]},
            "readings_compared": len(ca), "readings_changed": changed,
            "predictions": {"asks": a["predictions"], "not_executed": b["predictions"]}}


def trigger(model):
    return {run: {"asks": est(r["paired"]["all"]["ask_present_minus_absent"]),
                  "not_executed": -est(r["paired"]["all"]["exec_present_minus_absent"])} for run, r in pick(f"trigger_{model}")["runs"].items()}


def scale():
    effect = json.loads((RESULTS / "scale_qwen35_27b_main_dev.json").read_text())["rule_effect"]
    return {c: {"asks": est(v["ask_C_minus_N"]), "executes": est(v["exec_C_minus_N"])} for c, v in effect.items()}


def rules(model):
    cond = json.loads((RESULTS / f"rules_{model}_v2.json").read_text())["conditions"]
    return {c: {"asks": v["DP_asks"][0], "not_executed": round(1 - v["DP_executed"][0], 4)} for c, v in cond.items()}


def modes(model):
    out = {}
    for split, s in pick(f"modes_{model}")["splits"].items():
        for mode, r in s["modes"].items():
            for c, v in r["cells"].items():
                n = sum(v["labels"].values())
                if n:
                    out[f"{split}.{mode}.{c}"] = {"asks": round(v["labels"].get("ASK", 0) / n, 4),
                                                  "not_executed": round(1 - v["labels"].get("EXEC_MATCH", 0) / n, 4)}
    return out


def main():
    out = {"measures": {"asks": "evaluator label ASK", "not_executed": "any evaluator label other than EXEC_MATCH"},
           "assist": {m: assist(m) for m in ASSIST}, "trigger": {m: trigger(m) for m in TRIGGER}, "scale_27b_dev": scale(),
           "rules_dev": {m: rules(m) for m in RULES}, "modes": {m: modes(m) for m in MODES}}
    (RESULTS / "label_robustness.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({m: {"compared": v["readings_compared"], "changed": v["readings_changed"]} for m, v in out["assist"].items()}, indent=1))


if __name__ == "__main__":
    main()
