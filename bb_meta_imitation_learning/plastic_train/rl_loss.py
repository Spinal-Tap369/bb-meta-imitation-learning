# plastic_train/rl_loss.py

import torch
import torch.nn.functional as F

from .utils import PAD_ACTION
from .utils import smoothed_cross_entropy 

@torch.no_grad()
def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    T = rewards.shape[0]
    ret = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        ret[t] = running
    return ret

@torch.no_grad()
def mean_kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p_logp = F.log_softmax(p_logits, dim=-1)
    q_logp = F.log_softmax(q_logits, dim=-1)
    p = p_logp.exp()
    kl = (p * (p_logp - q_logp)).sum(dim=-1)
    return kl.mean()

@torch.no_grad()
def ess_ratio_from_rhos(rhos: torch.Tensor) -> torch.Tensor:
    s1 = rhos.sum()
    s2 = (rhos * rhos).sum()
    ess = (s1 * s1) / s2.clamp_min(1e-12)
    return ess / rhos.shape[0]

@torch.no_grad()
def compute_bc_at_theta(policy_net, batch_obs: torch.Tensor, batch_lab: torch.Tensor, args) -> torch.Tensor:
    if hasattr(policy_net, "reset_plastic"):
        policy_net.reset_plastic(batch_size=1, device=batch_obs.device)
        policy_net.set_plastic(update_traces=False, modulators=None)
    logits, _ = policy_net(batch_obs)
    if getattr(args, "label_smoothing", 0.0) > 0.0:
        return smoothed_cross_entropy(logits, batch_lab, ignore_index=PAD_ACTION, smoothing=args.label_smoothing).detach()
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
    return ce(logits.reshape(-1, logits.size(-1)), batch_lab.reshape(-1)).detach()

@torch.no_grad()
def outer_pg_term(policy_net,
                  exp: torch.Tensor, acts: torch.Tensor, beh_logits: torch.Tensor,
                  rews: torch.Tensor, args) -> torch.Tensor:
    if hasattr(policy_net, "reset_plastic"):
        policy_net.reset_plastic(batch_size=1, device=exp.device)
        policy_net.set_plastic(update_traces=False, modulators=None)
    logits_theta, _ = policy_net(exp.unsqueeze(0))
    logits_theta = logits_theta[0]
    returns = discounted_returns(rews, args.gamma)
    T = returns.size(0)
    t = torch.arange(T, device=returns.device, dtype=returns.dtype)
    zt = (t - t.mean()) / t.std().clamp_min(1e-6)
    Fmat = torch.stack([torch.ones_like(zt), zt], dim=1)
    FT_F = Fmat.T @ Fmat + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
    w = torch.linalg.solve(FT_F, Fmat.T @ returns)
    baseline = (Fmat @ w)
    adv = returns - baseline
    adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    logp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.view(-1,1)).squeeze(1)
    logp_beh = torch.log_softmax(beh_logits,   dim=-1).gather(1, acts.view(-1,1)).squeeze(1)
    rho = torch.exp(logp_cur - logp_beh)
    if getattr(args, "is_clip_rho", 0.0) > 0:
        rho = torch.clamp(rho, max=args.is_clip_rho)
    return ( - (rho * logp_cur * adv_norm).mean() ).detach()
