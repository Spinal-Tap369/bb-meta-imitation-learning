# plastic_train/model.py

import os
import torch
import logging
from torch.optim import Adam
from .plastic import PlasticLinear, PlasticConv1d
from bb_meta_imitation_learning.snail_trpo.snail_model import SNAILPolicyValueNet
import torch.nn as nn

logger = logging.getLogger(__name__)

class SNAILWithOptionalPlastic(torch.nn.Module):
    """
    Wraps SNAILPolicyValueNet and optionally replaces the policy head
    with a plastic head (Linear or Conv1d(1x1), depending on what SNAIL uses).
    """
    def __init__(self, *, use_plastic_head: bool, plastic_rule: str, plastic_init_eta: float,
                 plastic_learn_eta: bool, **snail_kwargs):
        super().__init__()
        self.core = SNAILPolicyValueNet(**snail_kwargs)
        self.use_plastic = bool(use_plastic_head)

        if self.use_plastic:
            if not hasattr(self.core, "policy_head"):
                raise AttributeError("SNAILPolicyValueNet must expose .policy_head for plastic replacement.")
            ph = self.core.policy_head

            # --- Conv1d(k=1) head (your current SNAIL) ---
            if isinstance(ph, nn.Conv1d):
                if ph.kernel_size != (1,):
                    raise ValueError("Plastic head supports Conv1d with kernel_size=1 only.")
                plastic = PlasticConv1d(
                    in_channels=ph.in_channels,
                    out_channels=ph.out_channels,
                    kernel_size=1,
                    init_eta=plastic_init_eta,
                    learn_eta=plastic_learn_eta,
                    rule=plastic_rule,
                    bias=(ph.bias is not None),
                )
                # copy weights/bias (squeeze the 1x1 kernel to matrix)
                with torch.no_grad():
                    plastic.W.copy_(ph.weight.squeeze(-1))
                    if ph.bias is not None and plastic.bias is not None:
                        plastic.bias.copy_(ph.bias)
                self.core.policy_head = plastic

            # --- Linear head (if you ever switch) ---
            elif isinstance(ph, nn.Linear):
                plastic = PlasticLinear(
                    ph.in_features, ph.out_features,
                    init_eta=plastic_init_eta,
                    learn_eta=plastic_learn_eta,
                    rule=plastic_rule,
                    bias=(ph.bias is not None),
                )
                with torch.no_grad():
                    plastic.W.copy_(ph.weight)
                    if ph.bias is not None and plastic.bias is not None:
                        plastic.bias.copy_(ph.bias)
                self.core.policy_head = plastic
            else:
                raise TypeError(f"Unsupported policy_head type: {type(ph).__name__} (expected Conv1d(k=1) or Linear)")

    # --- Plastic control proxies ---
    def reset_plastic(self, batch_size: int, device=None):
        if self.use_plastic:
            self.core.policy_head.reset_traces(batch_size, device=device)

    def set_plastic(self, *, update_traces: bool, modulators: torch.Tensor | None):
        if self.use_plastic:
            # WARNING: keep modulators detached (you already do this in train.py)
            m = None if modulators is None else modulators.detach()
            self.core.policy_head.set_mode(update_traces=update_traces, modulators=m)

    # --- Model API passthrough ---
    def forward(self, seq):
        return self.core(seq)

    def act_single_step(self, seq_prefix):
        # default: do NOT update traces during eval act unless caller enabled updates
        return self.core.act_single_step(seq_prefix)

def build_model(
    *,
    action_dim=3,
    base_dim=256,
    policy_filters=32,
    policy_attn_dim=16,
    value_filters=16,
    seq_len=500,
    use_plastic_head=False,
    plastic_rule="oja",
    plastic_init_eta=0.1,
    plastic_learn_eta=False,
):
    return SNAILWithOptionalPlastic(
        use_plastic_head=use_plastic_head,
        plastic_rule=plastic_rule,
        plastic_init_eta=plastic_init_eta,
        plastic_learn_eta=plastic_learn_eta,
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
