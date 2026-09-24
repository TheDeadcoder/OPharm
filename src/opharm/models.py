import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
from transformers import AutoModelForCausalLM

from opharm.chat import pins

_EXPORTING = qwen35.is_torchdynamo_exporting


def set_gdn_solver(mode):
    if mode not in ("default", "loop"):
        raise ValueError(mode)
    qwen35.is_torchdynamo_exporting = (lambda: True) if mode == "loop" else _EXPORTING


def load_model(key, device="mps", dtype=torch.bfloat16, solver=None, local_files_only=True):
    set_gdn_solver(solver or ("loop" if device == "mps" else "default"))
    p = pins()[key]
    model = AutoModelForCausalLM.from_pretrained(
        p["repo"], revision=p["revision"], dtype=dtype, device_map=device, local_files_only=local_files_only
    )
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def last_logits(model, input_ids, attention_mask=None):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1].float()
