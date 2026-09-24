import torch

from opharm.interp import hooks
from opharm.run.decision import action_logodds


def forward_capture(model, ids, positions, opener_id, edits=()):
    points = list(range(len(hooks.resid_modules(model))))
    store = {}
    with torch.inference_mode(), hooks.hooked(*edits, hooks.capture(model, points, positions, store)):
        logits = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1).logits[0, -1]
    acts = torch.stack([store[p][0] for p in points], dim=1)
    return action_logodds(logits.float().cpu(), opener_id).item(), logits.float().cpu(), acts
