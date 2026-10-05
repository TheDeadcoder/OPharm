from collections import Counter

import opharm
import pytest

from opharm.bench import assist
from opharm.bench.prompts import CLOSING


@pytest.fixture(scope="module")
def parts():
    return assist.build()


def user(r):
    return "".join(t for _, t in r["user"])


def system(r):
    return "".join(t for _, t in r["system"])


def test_counts(parts):
    assert Counter(r["arm"] for r in parts["op"]) == {a: 428 for a in ("A0", "Airr", "A1", "A2", "Iask", "Ibal", "Iaskbal")}
    assert Counter(r["arm"] for r in parts["cj"]) == {a: 528 for a in ("A0", "Airr", "A1", "A2", "Ibal")}
    assert Counter((r["rule"], r["arm"]) for r in parts["think"]) == {("op", "A0"): 266, ("op", "A1"): 266, ("cj", "A0"): 328}
    assert len(parts["kn_nt"]) == 428 + 528 and all(r["question"] == "qapp" for r in parts["kn_nt"])
    assert len(parts["kn"]) == 4 * 428 + 3 * 528
    assert all(len({r["id"] for r in p}) == len(p) for p in parts.values())


def test_operational_truth(parts):
    for r in parts["op"]:
        required = r["cell"] == "DP"
        assert r["required"] == (True if r["arm"] in ("Iask", "Iaskbal") else required)
        if r["cell"] == "BP":
            assert r["bp_clean"]
        if r["arm"] == "A2":
            assert assist.VERDICT[required] in user(r)


def test_conjunction_truth(parts):
    for r in parts["cj"]:
        rm, qm = r["state"]
        assert (r["policy_who"] == r["user"][0][1].split(" from ")[1].split(" (")[0]) == (rm == "1")
        assert (r["policy_queue"] == r["ticket_queue"]) == (qm == "1")
        assert r["required"] == (r["state"] == "11")
        assert f"Queue: {r['ticket_queue']}\n" in user(r)
        if r["arm"] == "A2":
            assert assist.VERDICT[r["required"]] in user(r)


def test_facts_are_true_and_reversible(parts):
    for r in parts["op"]:
        if r["arm"] != "A1":
            continue
        u = user(r)
        for word in ("cannot be restored", "permanently", "irreversibly", "for good"):
            assert word not in u
        assert assist.ENVIRONMENT[r["env"]] in u and assist.ENVIRONMENT["S" if r["env"] == "P" else "P"] not in u
        other = "B" if r["target"] == "D" else "D"
        assert assist.IMPORTANT[other] not in u


def test_blocks_share_one_location(parts):
    for r in parts["op"] + parts["cj"]:
        u = user(r)
        if r["arm"] in ("Airr", "A1", "A2"):
            head, block, tail = u.rsplit("\n\n", 2)
            assert tail == CLOSING and (block.startswith("Context: ") or block.startswith("Policy check: "))
            assert head.split("\n")[-1].startswith(("On-call engineer:", "Queue:"))
        else:
            assert u.endswith("\n\n" + CLOSING)


def test_interface_arms_change_only_their_line(parts):
    base = {r["id"].rsplit(".", 1)[0]: system(r) for r in parts["op"] if r["arm"] == "A0"}
    for r in parts["op"]:
        key = r["id"].rsplit(".", 1)[0]
        if r["arm"] == "Ibal":
            assert system(r) == base[key].replace(assist.FORMAT, assist.BALANCED)
        elif r["arm"] == "Iask":
            assert assist.POLICY_ALL in system(r) and assist.FORMAT in system(r)
        elif r["arm"] == "Iaskbal":
            assert assist.POLICY_ALL in system(r) and assist.BALANCED in system(r) and assist.FORMAT not in system(r)
        else:
            assert system(r) == base[key]


def test_names_and_queues_balanced(parts):
    cj = [r for r in parts["cj"] if r["arm"] == "A0"]
    who = Counter((r["policy_who"], r["state"][0]) for r in cj)
    assert all(who[(w, "1")] > 0 and who[(w, "0")] > 0 for w, _ in who)
    queue = Counter((r["policy_queue"], r["state"][1]) for r in cj)
    assert all(queue[(q, "1")] > 0 and queue[(q, "0")] > 0 for q, _ in queue)


def test_unambiguous_verdict(parts):
    rows = assist.extra_arms()
    assert len(rows) == 428 + 528 and all(r["arm"] == "A2u" for r in rows)
    base = {r["id"].rsplit(".", 1)[0]: r for r in parts["op"] + parts["cj"] if r["arm"] == "A0"}
    for r in rows:
        head, block, tail = user(r).rsplit("\n\n", 2)
        assert tail == CLOSING and block == assist.VERDICT_U[r["required"]]
        b = base[r["id"].rsplit(".", 1)[0]]
        assert system(r) == system(b) and r["required"] == b["required"] and user(b) == f"{head}\n\n{CLOSING}"


def test_condition_arm(parts):
    rows = assist.extra_arms(arms=("A3",))
    assert len(rows) == 428 + 528 and all(r["arm"] == "A3" for r in rows)
    base = {r["id"].rsplit(".", 1)[0]: r for r in parts["op"] + parts["cj"] if r["arm"] == "A0"}
    for r in rows:
        head, block, tail = user(r).rsplit("\n\n", 2)
        want = (assist.CONDITION[r["required"]] if r["rule"] == "op"
                else assist.CONDITION_CJ[r["required"]].format(who=r["policy_who"], queue=r["policy_queue"]))
        assert tail == CLOSING and block == want
        b = base[r["id"].rsplit(".", 1)[0]]
        assert system(r) == system(b) and r["required"] == b["required"] and user(b) == f"{head}\n\n{CLOSING}"
    assert {r["cell"] for r in rows if r["rule"] == "op" and r["required"]} == {"DP"}
    assert {r["state"] for r in rows if r["rule"] == "cj" and r["required"]} == {"11"}


def test_replication(parts):
    rows = assist.replication()
    n = Counter(r["arm"] for r in rows)
    assert n["A0"] == n["A2u"] and set(n) == {"A0", "A2u"} and len({r["skeleton"] for r in rows}) == 126
    assert not {r["skeleton"] for r in rows} & {r["skeleton"] for r in assist.pilot()}
    assert all(r["split"] == "dev" and r["rule"] == "op" for r in rows)
    assert not {r["skeleton"] for r in rows} & {r["skeleton"] for r in parts["op"]}
    assert {r["template"] for r in rows} == {r["template"] for r in parts["op"]}
    for r in rows:
        if r["arm"] == "A2u":
            assert user(r).rsplit("\n\n", 2)[1] == assist.VERDICT_U[r["required"]]
        else:
            assert user(r).endswith("\n\n" + CLOSING)
