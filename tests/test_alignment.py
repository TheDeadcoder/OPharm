import pytest

from opharm.bench.align import aligned_pairs, pair_exclusion
from opharm.chat import load_tokenizer, render

SYSTEM = [(None, "You are the operations agent.")]


def user(env, host="db-07"):
    return [(None, f"Please run `df -h` on {host}.\n\nEnvironment:"), ("env", env), (None, "\n\nPlease handle this ticket.")]


@pytest.fixture(scope="module", params=["qwen35_4b", "qwen3_4b_2507"])
def tok(request):
    return load_tokenizer(request.param)


def test_aligned_pair_passes(tok):
    a, b = render(tok, SYSTEM, user(" production")), render(tok, SYSTEM, user(" staging"))
    assert pair_exclusion(a, b, "env") is None


def test_change_outside_span_is_excluded(tok):
    a, b = render(tok, SYSTEM, user(" production")), render(tok, SYSTEM, user(" staging", host="db-08"))
    assert pair_exclusion(a, b, "env") == "outside_span"


def test_length_mismatch_is_excluded(tok):
    a, b = render(tok, SYSTEM, user(" production")), render(tok, SYSTEM, user(" a sandbox environment"))
    assert pair_exclusion(a, b, "env") == "length"


def test_aligned_pairs_search(tok):
    hits = aligned_pairs([tok], ["prod"], ["stage", "stg"], "Host: {}-db-07\n")
    assert ("prod", "stage") in hits and ("prod", "stg") not in hits
