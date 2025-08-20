# plastic_train/es.py

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
def meta_objective_from_rollout(policy_net, ro, batch_obs, batch_lab, args, device,
                                use_is_inner: bool = False, is_clip_rho: float | None = None,
                                ess_min_ratio: float | None = None):
    """
    Given a (possibly perturbed) policy and its explore rollout 'ro', perform the same
    plastic adaptation you already do, then return BC loss at φ (a scalar).
    If use_is_inner=True, correct the inner adaptation with IS weights ρ_t computed
    from current policy vs ro.beh_logits; drop task (return NaN) if ESS/T is too low.
    """
    from .rl_loss import discounted_returns
    # unpack + move
    exp = ro.obs6.to(device); acts = ro.actions.to(device)
    rews = ro.rewards.to(device) * args.rew_scale
    beh  = ro.beh_logits.to(device)
    rews = torch.clamp(rews, -args.rew_clip, args.rew_clip)

    # optionally truncate explore window
    if args.adapt_trunc_T and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
        exp = exp[-args.adapt_trunc_T:]; acts = acts[-args.adapt_trunc_T:]; rews = rews[-args.adapt_trunc_T:]; beh = beh[-args.adapt_trunc_T:]

    # pass 1: advantages at θ to build modulators
    policy_net.reset_plastic(batch_size=1, device=device)
    policy_net.set_plastic(update_traces=False, modulators=None)
    logits_theta, _ = policy_net(exp.unsqueeze(0))
    logits_theta = logits_theta[0]

    # IS weights under current (possibly perturbed) policy vs behavior
    if use_is_inner:
        lp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
        lp_beh = torch.log_softmax(beh,           dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
        rho = torch.exp(lp_cur - lp_beh)
        if (is_clip_rho is not None) and (is_clip_rho > 0):
            rho = torch.clamp(rho, max=is_clip_rho)
        # simple ESS/T
        with torch.no_grad():
            w = rho / rho.sum().clamp_min(1e-8)
            ess = 1.0 / (w.pow(2).sum().clamp_min(1e-8))
            ess_ratio = float(ess / max(1, rho.numel()))
        if (ess_min_ratio is not None) and (ess_ratio < ess_min_ratio):
            return torch.tensor(float("nan"), device=device)
    else:
        rho = None

    from .rl_loss import discounted_returns
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
    if use_is_inner:
        # Modulate plasticity by IS weights (correct inner update under reuse)
        m_t = m_t * rho.detach()

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

import contextlib

class ParamStep:
    """
    Temporarily apply a parameter update (theta -> theta + delta) and auto-revert.
    """
    def __init__(self, named_params, deltas):
        self.ops = []
        for (n, p), d in zip(named_params.items(), deltas):
            if d is None:
                d = torch.zeros_like(p)
            self.ops.append((p, d))
    def __enter__(self):
        for p, d in self.ops:
            p.data.add_(d)
    def __exit__(self, exc_type, exc, tb):
        for p, d in self.ops:
            p.data.sub_(d)

def _compute_adv_and_logp(policy_net, exp, acts, beh, args, device):
    """
    Shared helper: run policy on explore states to get logp(a|s), advantages, and (optionally) IS weights.
    """
    from .rl_loss import discounted_returns
    policy_net.reset_plastic(batch_size=1, device=device)
    policy_net.set_plastic(update_traces=False, modulators=None)
    logits_theta, _ = policy_net(exp.unsqueeze(0))   # (1,T,A)
    logits_theta = logits_theta[0]
    # A-hat_t via discount + linear baseline (matches your train loop)
    rews = beh.new_zeros(exp.size(0))  # not used here; returned separately when needed
    return logits_theta

@torch.no_grad()
def _ess_ratio_from_rho(rho: torch.Tensor) -> float:
    w = rho / rho.sum().clamp_min(1e-8)
    ess = 1.0 / (w.pow(2).sum().clamp_min(1e-8))
    return float(ess / max(1, rho.numel()))

def _compute_advantage(exp, rewards, args):
    from .rl_loss import discounted_returns
    returns = discounted_returns(rewards, args.gamma)
    T = returns.size(0)
    t = torch.arange(T, device=returns.device, dtype=returns.dtype)
    zt = (t - t.mean()) / t.std().clamp_min(1e-6)
    F = torch.stack([torch.ones_like(zt), zt], dim=1)
    FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
    w = torch.linalg.solve(FT_F, F.T @ returns)
    baseline = (F @ w)
    adv = returns - baseline
    adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    return adv_norm

def _one_step_reinforce_update(policy_net, exp, acts, beh, rewards, args, device,
                               alpha: float, scope: str, use_is: bool, is_clip: float | None,
                               ess_min_ratio: float | None):
    """
    Compute a one-step REINFORCE gradient on a *subset* of params (scope) and
    return the delta tensors to apply ephemerally.
    """
    # Forward for logπ(a|s)
    policy_net.reset_plastic(batch_size=1, device=device)
    policy_net.set_plastic(update_traces=False, modulators=None)
    with torch.enable_grad():
        logits, _ = policy_net(exp.unsqueeze(0))        # (1, T, A)
        logits = logits[0]
        logp = torch.log_softmax(logits, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)

        # IS weights if reusing behavior data
        if use_is:
            lp_beh = torch.log_softmax(beh, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
            rho = torch.exp(logp.detach() - lp_beh)    # detach logp for rho to avoid second-order terms
            if is_clip and is_clip > 0:
                rho = torch.clamp(rho, max=is_clip)
            if ess_min_ratio is not None and _ess_ratio_from_rho(rho) < ess_min_ratio:
                return None  # signal "drop task" for this perturb eval
        else:
            rho = None

        # Advantages (normalized) with linear baseline
        adv_norm = _compute_advantage(exp, rewards, args)

        # J = - sum_t [ (rho_t *) logπ(a_t|s_t) * Â_t ]   (mean is fine, same scale)
        if rho is not None:
            per_t = -(rho * logp * adv_norm)
        else:
            per_t = -(logp * adv_norm)
        J = per_t.mean()

        # grads on a subset
        named = select_es_named_params(policy_net, scope)  # e.g., "head"
        params = list(named.values())
        grads = torch.autograd.grad(J, params, create_graph=False, retain_graph=False, allow_unused=False)

        # delta = -alpha * grad  (SGD step)
        deltas = [(-alpha) * (g if g is not None else torch.zeros_like(p)) for p, g in zip(params, grads)]

    return named, deltas

@torch.no_grad()
def meta_objective_with_inner_pg(policy_net, ro, batch_obs, batch_lab, args, device,
                                 alpha: float, scope: str, use_is: bool, is_clip_rho: float | None,
                                 ess_min_ratio: float | None):
    """
    ES fitness with a one-step REINFORCE inner update on a small param subset.
    No persistent change (ephemeral fast-weights), no outer backprop through this step.
    """
    # unpack rollout
    exp = ro.obs6.to(device)
    acts = ro.actions.to(device)
    rewards = torch.clamp(ro.rewards.to(device) * args.rew_scale, -args.rew_clip, args.rew_clip)
    beh = ro.beh_logits.to(device)

    # optional truncation (match train loop)
    if args.adapt_trunc_T and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
        exp = exp[-args.adapt_trunc_T:]; acts = acts[-args.adapt_trunc_T:]; rewards = rewards[-args.adapt_trunc_T:]; beh = beh[-args.adapt_trunc_T:]

    # compute one-step update deltas
    out = _one_step_reinforce_update(policy_net, exp, acts, beh, rewards, args, device,
                                     alpha=alpha, scope=scope, use_is=use_is,
                                     is_clip=is_clip_rho, ess_min_ratio=ess_min_ratio)
    if out is None:
        return torch.tensor(float("nan"), device=device)
    named_sub, deltas = out

    # apply deltas ephemerally, then evaluate BC at phi
    with ParamStep(named_sub, deltas):
        logits_phi, _ = policy_net(batch_obs)
        if args.label_smoothing > 0.0:
            from .utils import smoothed_cross_entropy, PAD_ACTION
            loss_bc_phi = smoothed_cross_entropy(logits_phi, batch_lab, ignore_index=PAD_ACTION, smoothing=args.label_smoothing)
        else:
            from .utils import PAD_ACTION
            ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
            loss_bc_phi = ce(logits_phi.reshape(-1, logits_phi.size(-1)), batch_lab.reshape(-1))

    return loss_bc_phi.detach()

