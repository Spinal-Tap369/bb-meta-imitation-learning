# mri_train/remap.py

from typing import Dict, List
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def remap_pretrained_state(src_sd: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Map checkpoint keys to the current model naming:
      - strip 'module.' prefix (DDP)
      - add 'core.' prefix when missing
      - action_head -> policy_head
      - value_head  -> critic_head (older name)
    Keep only keys that exist in the destination model AND match shape.
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    def _strip_module(k: str) -> str:
        return k[len("module."):] if k.startswith("module.") else k

    def _with_core(k: str) -> str:
        return k if k.startswith("core.") else f"core.{k}"

    def _variants(k: str) -> List[str]:
        base = _strip_module(k)
        cands = set()
        bases = [base, _with_core(base)]
        for b in bases:
            cands.add(b)
            cands.add(b.replace("action_head", "policy_head"))
            cands.add(b.replace("value_head", "critic_head"))
        return list(cands)

    out = {}
    hits = 0
    for k, v in src_sd.items():
        for cand in _variants(k):
            if cand in model_keys and model_sd[cand].shape == v.shape:
                out[cand] = v
                hits += 1
                break
    logger.info("[INIT][REMAPPER] mapped %d/%d checkpoint tensors into current model", hits, len(src_sd))
    return out
