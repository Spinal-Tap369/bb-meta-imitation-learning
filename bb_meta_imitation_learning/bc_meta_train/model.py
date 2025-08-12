# bc_meta_train/model.py

import os
import torch
import logging
from torch.optim import Adam
from bb_meta_imitation_learning.snail_trpo.snail_model import SNAILPolicyValueNet

logger = logging.getLogger(__name__)

def build_model(
    action_dim=3,
    base_dim=256,
    policy_filters=32,
    policy_attn_dim=16,
    value_filters=16,
    seq_len=500,  
):
    return SNAILPolicyValueNet(
        action_dim=action_dim,
        base_dim=base_dim,
        policy_filters=policy_filters,
        policy_attn_dim=policy_attn_dim,
        value_filters=value_filters,
        seq_len=seq_len,
    )

def make_optimizer(policy_net, lr, weight_decay):
    return Adam(policy_net.parameters(), lr=lr, weight_decay=weight_decay)

def save_checkpoint(policy_net, epoch, best_val_score, checkpoint_dir, load_dir):
    fn = f"bc_meta_ckpt_epoch{epoch}.pth"
    path = os.path.join(checkpoint_dir, fn)
    ckpt = {
        "policy_state_dict": policy_net.state_dict(),
        "epoch": epoch,
        "best_val_score": best_val_score,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(load_dir, "chkload.pth"))
    logger.info(f"[CHECKPOINT] saved {fn}")

def load_checkpoint(policy_net, load_dir, checkpoint_file=None):
    if checkpoint_file is None:
        checkpoint_file = os.path.join(load_dir, "chkload.pth")
    if os.path.isfile(checkpoint_file):
        logger.info(f"[CHECKPOINT] loading {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("best_val_score", float("inf"))
    logger.info("[CHECKPOINT] none found, starting from scratch")
    return 0, float("inf")
