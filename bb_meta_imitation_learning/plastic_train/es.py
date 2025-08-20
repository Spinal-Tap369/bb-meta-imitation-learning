# plastic_train/es.py

# bb_meta_imitation_learning/plastic_train/es.py
from collections import OrderedDict
import torch
import torch.nn as nn

def select_es_named_params(model: nn.Module, scope: str):
    """
    Returns an OrderedDict[name -> Parameter] of params to perturb.
    - head: policy head only
    - policy: everything except value/critic
    - all:   all trainable (incl. encoder), still excluding buffers
    """
    names_params = OrderedDict(model.named_parameters())
    def is_critic(n): ln = n.lower(); return ("value" in ln) or ("critic" in ln) or ("vf" in ln)
    def is_policy_head(n): ln = n.lower(); return ("policy_head" in ln) or ("action_head" in ln) or ("logits" in ln)

    if scope == "head":
        return OrderedDict((n,p) for n,p in names_params.items() if is_policy_head(n) and p.requires_grad)
    if scope == "policy":
        return OrderedDict((n,p) for n,p in names_params.items() if (not is_critic(n)) and p.requires_grad)
    # "all"
    return OrderedDict((n,p) for n,p in names_params.items() if p.requires_grad)

def sample_eps(named_params: OrderedDict[str, torch.nn.Parameter], mode: str):
    """Return dict[name -> epsilon tensor] (Gaussian for ES, ±1 for SPSA)."""
    out = {}
    for n, p in named_params.items():
        if mode == "spsa":
            out[n] = torch.empty_like(p).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
        else:
            out[n] = torch.randn_like(p)
    return out

class PerturbContext:
    """In-place θ <- θ + sign * sigma * eps  (auto-reverts on exit)."""
    def __init__(self, named_params, eps, sigma, sign):
        self.deltas = []
        for n, p in named_params.items():
            d = (sigma * (1.0 if sign > 0 else -1.0)) * eps[n]
            self.deltas.append((p, d))
    def __enter__(self):
        for p, d in self.deltas: p.data.add_(d)
    def __exit__(self, exc_type, exc, tb):
        for p, d in self.deltas: p.data.sub_(d)

@torch.no_grad()
def meta_objective_from_rollout(policy_net, ro, batch_obs, batch_lab, args, device):
    """
    Given a perturbed policy and its explore rollout 'ro', perform the same
    plastic adaptation you already do, then return BC loss at φ (a scalar).
    """
    from .rl_loss import discounted_returns
    # unpack + move
    exp = ro.obs6.to(device); acts = ro.actions.to(device)
    rews = ro.rewards.to(device) * args.rew_scale
    rews = torch.clamp(rews, -args.rew_clip, args.rew_clip)

    # optionally truncate explore window
    if args.adapt_trunc_T and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
        exp = exp[-args.adapt_trunc_T:]; acts = acts[-args.adapt_trunc_T:]; rews = rews[-args.adapt_trunc_T:]

    # pass 1: advantages at θ to build modulators
    policy_net.reset_plastic(batch_size=1, device=device)
    policy_net.set_plastic(update_traces=False, modulators=None)
    logits_theta, _ = policy_net(exp.unsqueeze(0))
    logits_theta = logits_theta[0]

    returns = discounted_returns(rews, args.gamma)
    # per-task linear baseline (same as in train loop)
    T = returns.size(0)
    t = torch.arange(T, device=returns.device, dtype=returns.dtype)
    zt = (t - t.mean()) / t.std().clamp_min(1e-6)
    F = torch.stack([torch.ones_like(zt), zt], dim=1)
    FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
    w = torch.linalg.solve(FT_F, F.T @ returns)
    baseline = F @ w
    adv = returns - baseline
    adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    m_t = adv_norm.clamp(-args.plastic_clip_mod, args.plastic_clip_mod)

    # pass 2: adapt traces with modulators, then evaluate BC at φ
    policy_net.reset_plastic(batch_size=1, device=device)
    policy_net.set_plastic(update_traces=True, modulators=m_t.unsqueeze(0))
    _ = policy_net(exp.unsqueeze(0))  # update fast weights
    policy_net.set_plastic(update_traces=False, modulators=None)

    logits_phi, _ = policy_net(batch_obs)  # batch_obs already on device
    if args.label_smoothing > 0.0:
        from .utils import smoothed_cross_entropy, PAD_ACTION
        loss_bc_phi = smoothed_cross_entropy(logits_phi, batch_lab, ignore_index=PAD_ACTION, smoothing=args.label_smoothing)
    else:
        from .utils import PAD_ACTION
        ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
        loss_bc_phi = ce(logits_phi.reshape(-1, logits_phi.size(-1)), batch_lab.reshape(-1))

    return loss_bc_phi.detach()
