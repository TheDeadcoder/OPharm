import torch


def _special(tok):
    return set(tok.all_special_tokens) | set(tok.get_added_vocab())


def opener_ids(tok):
    special = _special(tok)
    for t in ("<|tool_call>", "<tool_call>"):
        if t in special:
            return [tok.convert_tokens_to_ids(t)]
    if "<|python_tag|>" in special:
        return sorted({tok.convert_tokens_to_ids("<|python_tag|>")} | {tok.encode(s, add_special_tokens=False)[0] for s in ('{"', "{")})
    raise ValueError("no known tool-call opener in this tokenizer")


def action_logodds(logits, opener):
    logits = logits.float()
    ids = [opener] if isinstance(opener, int) else list(opener)
    other = logits.clone()
    other[..., ids] = float("-inf")
    return torch.logsumexp(logits[..., ids], dim=-1) - torch.logsumexp(other, dim=-1)
