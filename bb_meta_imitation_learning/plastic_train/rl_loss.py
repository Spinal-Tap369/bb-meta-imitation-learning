# plastic_train/rl_loss.py

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

@torch.no_grad()
def _log_probs_from_logits(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return log π(a|·) gathered at `actions`. Supports (T,A) or (B,T,A)."""
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, actions.unsqueeze(-1)).squeeze(-1)

def gae(returns: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """GAE(λ) advantages from returns and value predictions. All 1D (T,)."""
    with torch.no_grad():
        deltas = returns - values
        adv = torch.zeros_like(values)
        running = 0.0
        for t in reversed(range(values.shape[0])):
            running = deltas[t] + gamma * lam * running
            adv[t] = running
    return adv

def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute Monte Carlo returns. 1D (T,)."""
    with torch.no_grad():
        T = rewards.shape[0]
        ret = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            ret[t] = running
    return ret

def reinforce_with_baseline(
    cur_logits: torch.Tensor,          # (T, A)
    actions: torch.Tensor,             # (T,)
    rewards: torch.Tensor,             # (T,)
    values: torch.Tensor,              # (T,)
    gamma: float,
    lam: float,
    use_gae: bool,
    entropy_coef: float = 0.0,
    behavior_logits: Optional[torch.Tensor] = None,  # (T, A) if off-policy
    offpolicy: str = "none",                         # "none" | "is" | "vtrace"
    is_clip_rho: float = 1.0,
    normalize_adv: bool = True,
    value_clip: float = 0.0,                         # if >0, clip value target delta like PPO
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    REINFORCE + value baseline with optional off-policy correction and guards.
    Returns (loss, mean_entropy, mean|adv|).
    """
    # Basic
    logp_all = F.log_softmax(cur_logits, dim=-1)
    probs    = logp_all.exp()
    logp_cur = torch.gather(logp_all, -1, actions.unsqueeze(-1)).squeeze(-1)
    entropy  = -(probs * logp_all).sum(dim=-1).mean()

    # Returns & advantages
    returns = discounted_returns(rewards, gamma)
    if use_gae:
        adv = gae(returns, values.detach(), gamma, lam)
    else:
        adv = returns - values.detach()

    # Off-policy corrections when reusing trajectories
    if offpolicy != "none" and behavior_logits is not None:
        with torch.no_grad():
            logp_beh = _log_probs_from_logits(behavior_logits, actions)
            rho = torch.exp(logp_cur - logp_beh)
            if is_clip_rho is not None and is_clip_rho > 0:
                rho = torch.clamp(rho, max=is_clip_rho)
        if offpolicy == "is":
            adv = rho * adv
        elif offpolicy == "vtrace":
            with torch.no_grad():
                v = values.detach()
                boot = torch.cat([v[1:], v[-1:]])
                deltas = rho * (rewards + gamma * boot - v)
                vt = torch.zeros_like(v)
                running = v[-1]
                for t in reversed(range(v.shape[0])):
                    next_term = (running - (v[t + 1] if t < v.shape[0] - 1 else v[-1]))
                    running = v[t] + deltas[t] + gamma * lam * next_term
                    vt[t] = running
                adv = vt - v

    # Normalize (per rollout) + NaN/Inf guards
    if normalize_adv:
        std = adv.std().clamp_min(1e-8)
        adv = (adv - adv.mean()) / std
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    logp_cur = torch.nan_to_num(logp_cur, nan=0.0, posinf=0.0, neginf=0.0)

    # Policy gradient
    loss_pg  = -(logp_cur * adv.detach()).mean()

    # Value loss with optional clipping (stability)
    if value_clip and value_clip > 0.0:
        v_pred      = values
        v_target    = returns.detach()
        v_clipped   = v_pred + (v_target - v_pred).clamp(min=-value_clip, max=value_clip)
        loss_v      = 0.5 * torch.max((v_pred - v_target).pow(2), (v_clipped - v_target).pow(2)).mean()
    else:
        loss_v = 0.5 * (values - returns.detach()).pow(2).mean()

    loss_ent = -entropy_coef * entropy
    loss = loss_pg + loss_v + loss_ent
    return loss, entropy.detach(), adv.abs().mean().detach()

@torch.no_grad()
def mean_kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(p || q) for logits of shape (T, A)."""
    p_logp = F.log_softmax(p_logits, dim=-1)
    q_logp = F.log_softmax(q_logits, dim=-1)
    p = p_logp.exp()
    kl = (p * (p_logp - q_logp)).sum(dim=-1)
    return kl.mean()

@torch.no_grad()
def ess_ratio_from_rhos(rhos: torch.Tensor) -> torch.Tensor:
    """Effective sample size ratio ESS/T from importance weights rhos (T,)."""
    s1 = rhos.sum()
    s2 = (rhos * rhos).sum()
    ess = (s1 * s1) / s2.clamp_min(1e-12)
    return ess / rhos.shape[0]

