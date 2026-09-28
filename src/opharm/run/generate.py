import torch

from opharm.run.decision import _special

END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<turn|>")
CLOSERS = ("</tool_call>", "<tool_call|>")


def end_ids(tok):
    special = _special(tok)
    ids = {tok.convert_tokens_to_ids(t) for t in END_TOKENS if t in special}
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def pad_id(tok):
    if "<|endoftext|>" in _special(tok):
        return tok.convert_tokens_to_ids("<|endoftext|>")
    return tok.pad_token_id if tok.pad_token_id is not None else min(end_ids(tok))


def greedy(model, tok, id_lists, max_new=64, batch=8, extra_stops=()):
    special = _special(tok)
    ends, pad = end_ids(tok), pad_id(tok)
    stops = sorted(ends | {tok.convert_tokens_to_ids(s) for s in extra_stops if s in special})
    order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
    texts = [None] * len(id_lists)
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            n = max(len(id_lists[i]) for i in idx)
            x = torch.tensor([[pad] * (n - len(id_lists[i])) + id_lists[i] for i in idx], device=model.device)
            mask = torch.tensor([[0] * (n - len(id_lists[i])) + [1] * len(id_lists[i]) for i in idx], device=model.device)
            out = model.generate(x, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=stops, pad_token_id=pad)
            for i, row in zip(idx, out[:, n:].tolist()):
                cut = next((k for k, t in enumerate(row) if t in ends or t == pad), len(row))
                texts[i] = tok.decode(row[:cut])
    return texts
