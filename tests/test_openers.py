import pytest
import torch

from opharm.chat import load_tokenizer, pins
from opharm.run.decision import action_logodds, opener_ids
from opharm.run.generate import end_ids, pad_id


def test_logodds_list_matches_single_id():
    logits = torch.randn(50)
    assert torch.allclose(action_logodds(logits, 7), action_logodds(logits, [7]))


@pytest.mark.parametrize("key,opener,end", [
    ("qwen35_4b", ["<tool_call>"], "<|im_end|>"),
    ("qwen3_4b_2507", ["<tool_call>"], "<|im_end|>"),
    ("llama31_8b", ["<|python_tag|>"], "<|eot_id|>"),
    ("gemma4_e4b", ["<|tool_call>"], "<turn|>"),
])
def test_openers_and_stops(key, opener, end):
    if key not in pins():
        pytest.skip("model not pinned")
    try:
        tok = load_tokenizer(key)
    except OSError:
        pytest.skip("tokenizer not downloaded")
    ids = opener_ids(tok)
    assert all(tok.convert_tokens_to_ids(t) in ids for t in opener)
    assert tok.convert_tokens_to_ids(end) in end_ids(tok)
    assert pad_id(tok) is not None
