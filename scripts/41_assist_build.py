import hashlib
import json
from collections import Counter

import opharm

from opharm.bench import assist
from opharm.paths import BENCH, RESULTS

OUT = BENCH / "assist"


def write(name, rows):
    path = OUT / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    parts = assist.build()
    parts["pilot"] = assist.pilot()
    parts["a2u"] = assist.posthoc()
    files = {name: {"sha256": write(name, rows), "n": len(rows),
                    "by_arm": dict(Counter(f"{r['rule']}.{r['arm']}" for r in rows))} for name, rows in parts.items()}
    sks = assist.cohort()
    m = assist.marks()
    info = {
        "files": files,
        "cohort": {"split": "heldout", "skeletons": len(sks), "templates": sorted({s["template"] for s in sks}),
                   "bp_clean_templates": sorted({s["template"] for s in sks if assist.bp_clean(m[s["template"]])}),
                   "think_skeletons": len(assist.think_skeletons(sks))},
        "wording": {"verdict": assist.VERDICT, "verdict_unambiguous_posthoc": assist.VERDICT_U, "important": assist.IMPORTANT, "environment": assist.ENVIRONMENT,
                    "balanced": assist.BALANCED, "policy_all": assist.POLICY_ALL, "policy_conj": assist.POLICY_CONJ,
                    "questions": assist.QUESTIONS, "queues": assist.QUEUES, "verbs": assist.VERB, "outage": assist.OUTAGE},
        "sources": {"instances": hashlib.sha256((BENCH / "instances.jsonl").read_bytes()).hexdigest(),
                    "skeletons": hashlib.sha256((BENCH / "skeletons.jsonl").read_bytes()).hexdigest()},
    }
    (RESULTS / "assist_instances.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: (v["n"], v["by_arm"]) for k, v in files.items()}, indent=1))


if __name__ == "__main__":
    main()
