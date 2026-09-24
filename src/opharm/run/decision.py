import torch


def action_logodds(logits, opener_id):
    logits = logits.float()
    other = logits.clone()
    other[..., opener_id] = float("-inf")
    return logits[..., opener_id] - torch.logsumexp(other, dim=-1)
