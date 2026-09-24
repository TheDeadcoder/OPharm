import torch


def greedy(model, tok, id_lists, max_new=64, batch=8, extra_stops=()):
    pad, end = tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_end|>")
    stops = [end] + [tok.convert_tokens_to_ids(s) for s in extra_stops]
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
                cut = next((k for k, t in enumerate(row) if t in (end, pad)), len(row))
                texts[i] = tok.decode(row[:cut])
    return texts
