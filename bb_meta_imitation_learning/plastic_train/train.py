# plastic_train/train.py

import os
import sys
import json
import math
import random
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from bb_meta_imitation_learning.env.maze_task import MazeTaskManager

from .config import parse_args
from .model import build_model, save_checkpoint, load_checkpoint
from .utils import (
    make_task_id,
    load_all_manifests,
    eval_sampled_val,
    SEQ_LEN,
    PAD_ACTION,
    smoothed_cross_entropy,
)
from .data import (
    concat_explore_and_exploit,
    first_demo_paths,
    assert_start_goal_match,
    load_phase2_six_and_labels,
    maybe_augment_demo_six_cpu,
)
from .explore import ExploreRollout, collect_explore_vec, make_base_env
from .rl_loss import mean_kl_logits, ess_ratio_from_rhos, discounted_returns

# ---- ES helpers ----
from .es import (
    select_es_named_params,
    sample_eps,
    PerturbContext,
    meta_objective_from_rollout,
    meta_objective_with_inner_pg,  # inner RL + post-adapt BC
)

logger = logging.getLogger(__name__)

# ---------- logging setup ----------
def _setup_logging(log_file: Optional[str], level: str):
    """Set up root logger to both stdout and an optional file."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt_console = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt_console)
    root.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fileh = logging.FileHandler(log_file, mode="a")
        fileh.setFormatter(fmt_file)
        root.addHandler(fileh)

    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)
    logging.getLogger(__name__).setLevel(lvl)


def _cuda_mem(prefix: str, device):
    if device.type != "cuda":
        return "CPU"
    a = torch.cuda.memory_allocated() / (1024**2)
    r = torch.cuda.memory_reserved() / (1024**2)
    return f"{prefix} mem: alloc={a:.1f}MB reserved={r:.1f}MB"


def _shape_str(t):
    if isinstance(t, torch.Tensor):
        return f"{tuple(t.shape)} {str(t.dtype).replace('torch.','')}"
    return str(type(t))


def _count_params(mod: nn.Module):
    tot = sum(p.numel() for p in mod.parameters())
    train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return tot, train


# ---------- optimizer helpers ----------
def _find_encoder_module(net: torch.nn.Module):
    cand = ["image_encoder", "cnn", "conv_frontend", "visual_encoder", "encoder", "backbone", "combined_encoder"]
    root = net.core if hasattr(net, "core") else net
    for name in cand:
        mod = getattr(root, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod, name
    for name, mod in root.named_children():
        lname = name.lower()
        if any(k in lname for k in ["enc", "cnn", "conv", "vision", "image"]) and isinstance(mod, torch.nn.Module):
            return mod, name
    return None, None


def _critic_param_names(net: nn.Module):
    names = []
    for n, _ in net.named_parameters():
        ln = n.lower()
        if ("value" in ln) or ("critic" in ln) or ("vf" in ln):
            names.append(n)
    return set(names)


# ---------- small helpers used in both paths ----------
@torch.no_grad()
def _compute_bc_theta(policy_net: nn.Module, batch_obs: torch.Tensor, batch_lab: torch.Tensor, args) -> torch.Tensor:
    """BC loss at θ (no plastic) for Δ gate / monitoring."""
    if hasattr(policy_net, "reset_plastic"):
        policy_net.reset_plastic(batch_size=1, device=batch_obs.device)
        policy_net.set_plastic(update_traces=False, modulators=None)
    logits_bc_theta, _ = policy_net(batch_obs)
    if args.label_smoothing > 0.0:
        return smoothed_cross_entropy(logits_bc_theta, batch_lab, ignore_index=PAD_ACTION, smoothing=args.label_smoothing).detach()
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
    return ce(logits_bc_theta.reshape(-1, logits_bc_theta.size(-1)), batch_lab.reshape(-1)).detach()


@torch.no_grad()
def _outer_pg_term(policy_net: nn.Module,
                   exp: torch.Tensor, acts: torch.Tensor, beh_logits: torch.Tensor,
                   rews: torch.Tensor, args) -> torch.Tensor:
    """
    Monitor-only: outer PG at θ with IS clip. Not used in loss.
    """
    if hasattr(policy_net, "reset_plastic"):
        policy_net.reset_plastic(batch_size=1, device=exp.device)
        policy_net.set_plastic(update_traces=False, modulators=None)
    logits_theta, _ = policy_net(exp.unsqueeze(0))
    logits_theta = logits_theta[0]
    returns = discounted_returns(rews, args.gamma)
    # per-task linear baseline
    T = returns.size(0)
    t = torch.arange(T, device=returns.device, dtype=returns.dtype)
    zt = (t - t.mean()) / t.std().clamp_min(1e-6)
    F = torch.stack([torch.ones_like(zt), zt], dim=1)
    FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
    w = torch.linalg.solve(FT_F, F.T @ returns)
    baseline = (F @ w)
    adv = returns - baseline
    adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

    logp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
    logp_beh = torch.log_softmax(beh_logits, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
    rho = torch.exp(logp_cur - logp_beh)
    if args.is_clip_rho and args.is_clip_rho > 0:
        rho = torch.clamp(rho, max=args.is_clip_rho)
    return ( - (rho * logp_cur * adv_norm).mean() ).detach()


# ---------- util: concise ID printing ----------
def _fmt_ids(ids: List[int], limit: int = 8) -> str:
    ids = list(ids)
    if len(ids) <= limit:
        return "[" + ",".join(str(x) for x in ids) + "]"
    return "[" + ",".join(str(x) for x in ids[:limit]) + ",...;n=" + str(len(ids)) + "]"


# ---------- vectorized recollect helper ----------
def _recollect_batch_for_sign(
    policy_net: nn.Module,
    per_task_tensors: Dict[int, Dict[str, torch.Tensor]],
    tasks_used: List[int],
    device: torch.device,
    seed_base: int,
    vec_cap: int,
    dbg: bool = False,
    dbg_timing: bool = False,
    dbg_level: str = "WARNING",
    # new: force INFO logging for ES inner recollects (even if not in DEBUG)
    log_info: bool = False,
    sign_label: str = "",
) -> Dict[int, ExploreRollout]:
    """
    Recollect phase-1 rollouts for a set of tasks using vectorized envs.

    We keep CRN by using the *same* seed_base for the +/- pair, and a deterministic
    offset per chunk (off) so that the mapping is stable across both signs.

    Returns:
        dict {tid -> ExploreRollout}
    """
    ros_by_tid: Dict[int, ExploreRollout] = {}
    if not tasks_used:
        return ros_by_tid

    all_cfgs = [MazeTaskManager.TaskConfig(**per_task_tensors[tid]["task_dict"]) for tid in tasks_used]
    step = max(1, int(vec_cap))

    for off in range(0, len(all_cfgs), step):
        cfgs = all_cfgs[off: off + step]
        slice_tids = tasks_used[off: off + step]

        if log_info:
            logger.info(
                "[COLLECT][ES%s] chunk=%d..%d/%d vec_cap=%d seed_base=%d tasks=%s",
                sign_label, off, off + len(cfgs) - 1, len(all_cfgs) - 1, step, seed_base + off, _fmt_ids(slice_tids)
            )

        ro_list = collect_explore_vec(
            policy_net, cfgs, device, max_steps=250,
            seed_base=seed_base + off,
            dbg=dbg, dbg_timing=dbg_timing, dbg_level=dbg_level
        )

        for tid, ro in zip(slice_tids, ro_list):
            ros_by_tid[tid] = ro

    return ros_by_tid


# ---------- training loop ----------
def run_training():
    from .remap import remap_pretrained_state  # local import to keep module edges clean

    args = parse_args()

    # Prefer explicit --log_level; else --debug_level; else DEBUG if --debug else INFO.
    chosen_level = (args.log_level or args.debug_level or ("DEBUG" if args.debug else "INFO")).upper()
    _setup_logging(log_file=args.log_file, level=chosen_level)

    # Debug flags (control *what* we log; chosen_level controls verbosity)
    debug = bool(getattr(args, "debug", False))
    debug_level = str(getattr(args, "debug_level", "INFO")).upper()
    debug_every_batches = int(getattr(args, "debug_every_batches", 1))
    debug_tasks_per_batch = int(getattr(args, "debug_tasks_per_batch", 4))
    debug_inner_per_task = bool(getattr(args, "debug_inner_per_task", False))
    debug_mem = bool(getattr(args, "debug_mem", False))
    debug_timing = bool(getattr(args, "debug_timing", False))
    debug_shapes = bool(getattr(args, "debug_shapes", False))
    log_every_step = bool(getattr(args, "log_every_step", False))

    # Seeding
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("[SETUP] seed=%d", args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.load_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.info("[DEVICE] device=%s amp=%s %s", device.type, str(use_amp), _cuda_mem("init", device))

    # Load tasks
    with open(args.train_trials) as f:
        tasks_all = json.load(f)
    task_index_to_dict = {i: t for i, t in enumerate(tasks_all)}
    task_hash_to_dict = {make_task_id(t): t for t in tasks_all}
    logger.info("[DATA] loaded train_trials=%s num_tasks=%d", args.train_trials, len(tasks_all))

    # Load demo manifests
    demos_by_task = load_all_manifests(args.demo_root)
    if not demos_by_task:
        raise RuntimeError("No demo manifests found in demo_root.")
    logger.info("[DATA] demo_root=%s tasks_with_demos=%d", args.demo_root, len(demos_by_task))

    # Build task entries
    tasks = []
    for tid, recs in demos_by_task.items():
        if len(recs) < 1:
            continue
        if tid in task_index_to_dict:
            tdict = task_index_to_dict[tid]
        elif tid in task_hash_to_dict:
            tdict = task_hash_to_dict[tid]
        else:
            continue
        assert_start_goal_match(recs, tdict, tid)
        p2_paths = first_demo_paths(recs, args.demo_root)
        if len(p2_paths) == 0:
            continue
        tasks.append({"task_id": tid, "task_dict": tdict, "p2_paths": p2_paths})

    if not tasks:
        raise RuntimeError("No tasks with phase-2 demos found.")
    random.shuffle(tasks)
    n_val = min(len(tasks), args.val_size)
    val_tasks = tasks[:n_val]
    train_tasks = tasks[n_val:]
    logger.info("[SPLIT] train=%d val=%d (val_size=%d)", len(train_tasks), len(val_tasks), n_val)

    # Model (plastic head only)
    policy_net = build_model(
        seq_len=SEQ_LEN,
        use_plastic_head=True,
        plastic_rule=getattr(args, "plastic_rule", "oja"),
        plastic_init_eta=getattr(args, "plastic_eta", 0.1),
        plastic_learn_eta=bool(getattr(args, "plastic_learn_eta", False)),
    ).to(device)
    tot, trn = _count_params(policy_net)
    logger.info("[MODEL] built model params total=%d trainable=%d", tot, trn)

    # ---------- Load BC init + remap (no extra checks) ----------
    if args.bc_init:
        ck_path = os.path.abspath(args.bc_init)
        if not os.path.isfile(ck_path):
            raise FileNotFoundError(f"{ck_path} not found")
        raw_sd = torch.load(ck_path, map_location="cpu")
        remap_sd = remap_pretrained_state(raw_sd, policy_net)
        ret = policy_net.load_state_dict(remap_sd, strict=False)
        miss = getattr(ret, "missing_keys", []); unex = getattr(ret, "unexpected_keys", [])
        logger.info(f"[INIT] loaded BC init (after remap) from {ck_path} "
                    f"(matched={len(remap_sd)} missing={len(miss)} unexpected={len(unex)})")
        logger.info("[INIT] plastic head fast weights (H, _eta_const/param) missing in ckpt is expected.")
    else:
        logger.info("[INIT] No BC init provided.")

    # -------- optimizer(s): encoder warmup + separate critic head --------
    encoder_module, enc_name = _find_encoder_module(policy_net)
    critic_names = _critic_param_names(policy_net)

    all_named_params = dict(policy_net.named_parameters())
    critic_params = [all_named_params[n] for n in critic_names]
    non_critic_params = [p for n,p in all_named_params.items() if n not in critic_names]

    warmup_epochs = max(0, int(args.freeze_encoder_warmup_epochs))
    if encoder_module is None:
        logger.warning("No visual encoder detected; single param group (excluding critic head).")
        optimizer = torch.optim.Adam([p for p in non_critic_params], lr=args.lr, weight_decay=args.weight_decay)
        have_encoder = False
    else:
        have_encoder = True
        logger.info(f"[ENCODER] Using submodule '{enc_name}'.")
        enc_params = list(encoder_module.parameters())
        enc_ids = {id(p) for p in enc_params}
        rest_params = [p for p in non_critic_params if id(p) not in enc_ids]
        if warmup_epochs > 0:
            for p in encoder_module.parameters(): p.requires_grad = False
            optimizer = torch.optim.Adam(rest_params, lr=args.lr, weight_decay=args.weight_decay)
            logger.info(f"[ENCODER] Frozen for first {warmup_epochs} epoch(s).")
        else:
            for p in encoder_module.parameters(): p.requires_grad = True
            optimizer = torch.optim.Adam(
                [
                    {"params": rest_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": enc_params,  "lr": args.lr * args.encoder_lr_mult, "weight_decay": args.weight_decay},
                ]
            )
            logger.info(f"[ENCODER] No warmup; encoder lr mult = {args.encoder_lr_mult}.")

    # separate critic-only optimizer (optional aux)
    if len(critic_params) == 0:
        logger.warning("[CRITIC] No params matched value/critic head names; critic aux updates will be skipped.")
        opt_critic = None
    else:
        opt_critic = torch.optim.Adam(critic_params, lr=args.lr * args.critic_lr_mult, weight_decay=args.weight_decay)
        logger.info("[CRITIC] Critic head params=%d (lr x%.2f)", sum(p.numel() for p in critic_params), args.critic_lr_mult)

    # Resume
    start_epoch, best_val_score = load_checkpoint(policy_net, args.load_path)
    best_epoch = start_epoch
    logger.info("[RESUME] from epoch %d", start_epoch + 1)

    # Val env
    val_env = make_base_env()
    val_env.action_space.seed(args.seed)

    patience = 0
    final_epoch = start_epoch

    free_every = 8
    updates_since_free = 0
    if debug_shapes:
        logger.info("[SHAPES] SEQ_LEN=%s", str(SEQ_LEN))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        final_epoch = epoch
        logger.info("========== [EPOCH %02d/%02d] ==========", epoch, args.epochs)

        if have_encoder and warmup_epochs > 0 and epoch == (warmup_epochs + 1):
            for p in encoder_module.parameters(): p.requires_grad = True
            enc_params = list(encoder_module.parameters())
            enc_ids = {id(p) for p in enc_params}
            rest_params = [p for n,p in all_named_params.items() if (id(p) not in enc_ids) and (n not in critic_names)]
            optimizer = torch.optim.Adam(
                [
                    {"params": rest_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": enc_params,  "lr": args.lr * args.encoder_lr_mult, "weight_decay": args.weight_decay},
                ]
            )
            logger.info("[ENCODER] Warmup complete; switched to two param groups.")

        explore_cache: Dict[int, ExploreRollout] = {}
        num_batches = math.ceil(len(train_tasks) / max(1, args.batch_size))
        batch_indices = list(range(len(train_tasks)))
        random.shuffle(batch_indices)
        logger.info("[TRAIN] num_batches=%d batch_size=%d", num_batches, args.batch_size)

        policy_net.train()
        running_bc = 0.0
        running_inrl = 0.0
        count_updates = 0

        pbar = tqdm(range(num_batches), desc=f"[Epoch {epoch:02d}] train", leave=False, disable=getattr(args, "no_tqdm", False))
        for b in pbar:
            # Only show per-task chatter at DEBUG
            is_debug_batch = debug and ((b % max(1, debug_every_batches)) == 0)
            is_task_verbose = is_debug_batch and (debug_level == "DEBUG")

            start = b * args.batch_size
            end = min(len(train_tasks), (b + 1) * args.batch_size)
            batch_ids = [batch_indices[i] for i in range(start, end)]
            batch_tasks = [train_tasks[i] for i in batch_ids]

            if is_task_verbose:
                tids = [t["task_id"] for t in batch_tasks]
                logger.info("[BATCH %d/%d] tasks=%s", b+1, num_batches, _fmt_ids(tids))
                if debug_mem:
                    logger.info("[MEM][BATCH-START] %s", _cuda_mem("batch-start", device))

            # Fresh explores where needed
            need_collect = []
            for task in batch_tasks:
                tid = task["task_id"]
                if tid not in explore_cache or explore_cache[tid].reuse_count >= args.explore_reuse_M:
                    need_collect.append(task)

            _vec_cap = int(getattr(args, "num_envs", 8))
            if need_collect:
                logger.info("[COLLECT][BASE] epoch=%d batch=%d vec_cap=%d seed_base=%d tasks=%s",
                            epoch, b+1, _vec_cap,
                            args.seed + 100000 * epoch + 1000 * b,
                            _fmt_ids([t["task_id"] for t in need_collect]))

            for off in range(0, len(need_collect), max(1, _vec_cap)):
                slice_tasks = need_collect[off: off + max(1, _vec_cap)]
                cfgs = [MazeTaskManager.TaskConfig(**t["task_dict"]) for t in slice_tasks]
                seed_here = args.seed + 100000 * epoch + 1000 * b + off

                if slice_tasks:
                    logger.info("[COLLECT][BASE] chunk off=%d n=%d seed=%d tasks=%s",
                                off, len(slice_tasks), seed_here, _fmt_ids([t["task_id"] for t in slice_tasks]))

                ro_list = collect_explore_vec(
                    policy_net, cfgs, device, max_steps=250,
                    seed_base=seed_here,
                    dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                )
                for ttask, ro in zip(slice_tasks, ro_list):
                    explore_cache[ttask["task_id"]] = ro
                    if is_task_verbose and debug_shapes:
                        logger.debug("[COLLECT] tid=%s Tx=%d obs6=%s actions=%s rewards=%s",
                                     ttask["task_id"], ro.obs6.shape[0], _shape_str(ro.obs6),
                                     _shape_str(ro.actions), _shape_str(ro.rewards))

            # Build per-task tensors (once)
            per_task_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
            tasks_used: List[int] = []

            for idx_task, task in enumerate(batch_tasks):
                tid = task["task_id"]
                cfg = MazeTaskManager.TaskConfig(**task["task_dict"])
                ro = explore_cache.get(tid, None)
                if ro is None:
                    seed_here = args.seed + 424242
                    logger.info("[COLLECT][BASE-MISS] tid=%s seed=%d", tid, seed_here)
                    ro = collect_explore_vec(
                        policy_net, [cfg], device, max_steps=250,
                        seed_base=seed_here,
                        dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                    ).pop()
                    explore_cache[tid] = ro

                exp_six_dev  = ro.obs6.to(device, non_blocking=True)
                actions_x    = ro.actions.to(device, non_blocking=True)
                rewards_x    = ro.rewards.to(device, non_blocking=True)
                beh_logits_x = ro.beh_logits.to(device, non_blocking=True)
                Tx = exp_six_dev.shape[0]

                # KL refresh guard
                with torch.no_grad():
                    logits_now_tmp, _ = policy_net(exp_six_dev.unsqueeze(0))
                    logits_now_tmp = logits_now_tmp[0] if logits_now_tmp.dim() == 3 else logits_now_tmp
                    kl_val = mean_kl_logits(logits_now_tmp, beh_logits_x).item()
                if is_task_verbose:
                    logger.info("[KL] tid=%s mean_kl=%.4f thr=%.4f", tid, kl_val, args.kl_refresh_threshold)
                if kl_val > args.kl_refresh_threshold:
                    seed_here = args.seed + 999999
                    logger.info("[COLLECT][KL-REFRESH] tid=%s seed=%d (kl=%.4f > %.4f)", tid, seed_here, kl_val, args.kl_refresh_threshold)
                    ro = collect_explore_vec(
                        policy_net, [cfg], device, max_steps=250,
                        seed_base=seed_here,
                        dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                    ).pop()
                    explore_cache[tid] = ro
                    exp_six_dev  = ro.obs6.to(device, non_blocking=True)
                    actions_x    = ro.actions.to(device, non_blocking=True)
                    rewards_x    = ro.rewards.to(device, non_blocking=True)
                    beh_logits_x = ro.beh_logits.to(device, non_blocking=True)
                    Tx = exp_six_dev.shape[0]
                    logger.info("[KL][REFRESH] tid=%s new_Tx=%d", tid, Tx)

                # ESS guard (optional)
                if args.ess_refresh_ratio and args.ess_refresh_ratio > 0:
                    with torch.no_grad():
                        logits_now, _ = policy_net(exp_six_dev.unsqueeze(0))
                        logits_now = logits_now[0] if logits_now.dim() == 3 else logits_now
                        lp_cur = torch.log_softmax(logits_now, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                        lp_beh = torch.log_softmax(beh_logits_x, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                        rhos = torch.exp(lp_cur - lp_beh)
                        ess_ratio = ess_ratio_from_rhos(rhos).item()
                    if is_task_verbose:
                        logger.info("[ESS] tid=%s Tx=%d ess_ratio=%.3f thr=%.3f reuse_count=%d",
                                    tid, Tx, ess_ratio, args.ess_refresh_ratio, ro.reuse_count)
                    if int(getattr(args, "explore_reuse_M", 1)) > 1 and ess_ratio < args.ess_refresh_ratio:
                        seed_here = args.seed + 31337
                        logger.info("[COLLECT][ESS-REFRESH] tid=%s seed=%d (ess=%.3f < %.3f)",
                                    tid, seed_here, ess_ratio, args.ess_refresh_ratio)
                        ro = collect_explore_vec(
                            policy_net, [cfg], device, max_steps=250,
                            seed_base=seed_here,
                            dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                        ).pop()
                        explore_cache[tid] = ro
                        exp_six_dev  = ro.obs6.to(device, non_blocking=True)
                        actions_x    = ro.actions.to(device, non_blocking=True)
                        rewards_x    = ro.rewards.to(device, non_blocking=True)
                        beh_logits_x = ro.beh_logits.to(device, non_blocking=True)
                        Tx = exp_six_dev.shape[0]
                        logger.info("[ESS][REFRESH] tid=%s new_Tx=%d", tid, Tx)

                # Build batched BC tensors from ALL demos (once)
                prev_action_start = float(actions_x[-1].item()) if Tx > 0 else 0.0
                demo_obs_list: List[torch.Tensor] = []
                demo_lab_list: List[torch.Tensor] = []

                for demo_path in task["p2_paths"]:
                    p2_six, p2_labels = load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start)
                    if p2_six.numel() == 0:
                        continue
                    p2_six = maybe_augment_demo_six_cpu(p2_six, args)
                    p2_six = p2_six.to(device, non_blocking=True)
                    p2_labels = p2_labels.to(device, non_blocking=True)
                    obs6_cat, labels_cat = concat_explore_and_exploit(exp_six_dev, p2_six, p2_labels)
                    demo_obs_list.append(obs6_cat)
                    demo_lab_list.append(labels_cat)

                if len(demo_obs_list) == 0:
                    logger.warning("[BC][SKIP] tid=%s no demos to supervise in this batch", tid)
                    continue

                B_d = len(demo_obs_list)
                T_max = max(x.shape[0] for x in demo_obs_list)
                H, W = demo_obs_list[0].shape[-2], demo_obs_list[0].shape[-1]

                pad_obs = []
                pad_lab = []
                for x, y in zip(demo_obs_list, demo_lab_list):
                    t = x.shape[0]
                    if t < T_max:
                        pad_t = T_max - t
                        x = torch.cat([torch.zeros((pad_t, 6, H, W), device=device, dtype=x.dtype), x], dim=0)
                        y = torch.cat([torch.full((pad_t,), PAD_ACTION, device=device, dtype=y.dtype), y], dim=0)
                    pad_obs.append(x); pad_lab.append(y)

                batch_obs = torch.stack(pad_obs, dim=0)  # (B_d, T_max, 6,H,W)
                batch_lab = torch.stack(pad_lab, dim=0)  # (B_d, T_max)

                if is_task_verbose and debug_shapes:
                    logger.info("[BC][BUILD] tid=%s B_d=%d T_max=%d obs=%s lab=%s",
                                tid, B_d, T_max, _shape_str(batch_obs), _shape_str(batch_lab))

                per_task_tensors[tid] = {
                    "batch_obs": batch_obs,
                    "batch_lab": batch_lab,
                    "exp": exp_six_dev,
                    "actions": actions_x,
                    "rewards": rewards_x,
                    "beh_logits": beh_logits_x,
                    "ro": ro,
                    "task_dict": task["task_dict"],
                }
                tasks_used.append(tid)

            if debug and debug_tasks_per_batch > 0:
                tasks_used = tasks_used[:debug_tasks_per_batch]

            if len(tasks_used) == 0:
                continue

            if is_task_verbose and debug_mem:
                logger.info("[MEM][PRE-K] %s", _cuda_mem("pre-k", device))

            # -------- Critic auxiliary regression (optional) --------
            if opt_critic is not None and args.critic_aux_steps > 0:
                for s in range(args.critic_aux_steps):
                    opt_critic.zero_grad(set_to_none=True)
                    vlosses = []
                    for tid in tasks_used:
                        tensors = per_task_tensors[tid]
                        exp = tensors["exp"]
                        rews = tensors["rewards"] * args.rew_scale
                        rews = torch.clamp(rews, -args.rew_clip, args.rew_clip)
                        with autocast(device_type="cuda", enabled=use_amp):
                            _, values_x = policy_net(exp.unsqueeze(0))
                            values_x = values_x[0]
                            targets = discounted_returns(rews, args.gamma)
                            if args.critic_value_clip and args.critic_value_clip > 0.0:
                                v_pred    = values_x
                                v_target  = targets.detach()
                                v_clipped = v_pred + (v_target - v_pred).clamp(min=-args.critic_value_clip, max=args.critic_value_clip)
                                vloss     = 0.5 * torch.max((v_pred - v_target).pow(2), (v_clipped - v_target).pow(2)).mean()
                            else:
                                vloss = 0.5 * (values_x - targets.detach()).pow(2).mean()
                        vlosses.append(vloss)
                    if vlosses:
                        vloss = sum(vlosses) / float(len(vlosses))
                        vloss.backward()
                        torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
                        opt_critic.step()
                        if is_task_verbose:
                            logger.info("[CRITIC][AUX] step=%d/%d vloss=%.4f", s+1, args.critic_aux_steps, float(vloss.detach().cpu()))

            # =========================
            # OUTER LOOP
            # =========================
            for step_idx in range(args.nbc):

                # ----------- ES/SPSA branch (black-box outer) -----------
                if getattr(args, "es_enabled", False):
                    optimizer.zero_grad(set_to_none=True)

                    # Config for inner PG vs plasticity-based inner
                    use_pg_inner = (getattr(args, "es_inner_pg_alpha", 0.0) > 0.0)
                    inner_use_is = (not getattr(args, "es_recollect_inner", False)) and bool(getattr(args, "es_inner_pg_use_is", False))

                    # Compute f(θ) baseline (unperturbed after-adapt BC) for logging/baseline
                    with torch.no_grad():
                        bc_phi_base = []
                        for tid in tasks_used:
                            t = per_task_tensors[tid]
                            ro0 = t["ro"]
                            if use_pg_inner:
                                bc_phi0 = meta_objective_with_inner_pg(
                                    policy_net,
                                    ro0,
                                    t["batch_obs"], t["batch_lab"],
                                    args, device,
                                    alpha=getattr(args, "es_inner_pg_alpha", 0.0),
                                    scope=getattr(args, "es_inner_pg_scope", "head"),
                                    use_is=inner_use_is,
                                    is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                    ess_min_ratio=(getattr(args, "es_ess_min_ratio", None) if not getattr(args, "es_recollect_inner", False) else None),
                                )
                            else:
                                bc_phi0 = meta_objective_from_rollout(
                                    policy_net,
                                    ro0,
                                    t["batch_obs"], t["batch_lab"],
                                    args, device,
                                    use_is_inner=(not getattr(args, "es_recollect_inner", False)) and bool(getattr(args, "es_use_is_inner", False)),
                                    is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                    ess_min_ratio=(getattr(args, "es_ess_min_ratio", None) if not getattr(args, "es_recollect_inner", False) else None),
                                )
                            bc_phi_base.append(float(bc_phi0.cpu()))
                        if bc_phi_base:
                            running_bc += float(np.mean(bc_phi_base))
                            count_updates += 1
                    f_theta_baseline = float(np.mean(bc_phi_base)) if bc_phi_base else 0.0

                    # Prepare ES named params & grads
                    named_params = select_es_named_params(policy_net, getattr(args, "es_scope", "policy"))
                    if is_task_verbose:
                        logger.info("[ES] scope=%s popsize=%d sigma=%.4f params=%d",
                                    getattr(args, "es_scope", "policy"),
                                    int(getattr(args, "es_popsize", 8)),
                                    float(getattr(args, "es_sigma", 0.02)),
                                    len(named_params))

                    es_grads = {n: torch.zeros_like(p) for n, p in named_params.items()}
                    vec_cap = int(getattr(args, "num_envs", 8))  # capacity for vectorized recollects

                    # Population loop (antithetic)
                    for i in range(int(getattr(args, "es_popsize", 8))):
                        eps = sample_eps(named_params, "spsa" if getattr(args, "es_algo", "es") == "spsa" else "es")
                        # Common random numbers: fixed seed for the +/− pair
                        use_common = bool(getattr(args, "es_common_seed", False))
                        pair_seed = (args.seed + 17_000_000 * epoch + 10_000 * b + 100 * step_idx + i) if use_common else None

                        # -------- f(θ + σ ε)
                        with PerturbContext(named_params, eps, getattr(args, "es_sigma", 0.02), +1):
                            bc_list_plus = []
                            ros_plus_by_tid: Dict[int, ExploreRollout] = {}
                            if getattr(args, "es_recollect_inner", False):
                                seed_for_pair = (pair_seed if pair_seed is not None else args.seed + 99991 + i)
                                # Force INFO logs for trajectory collection
                                ros_plus_by_tid = _recollect_batch_for_sign(
                                    policy_net, per_task_tensors, tasks_used, device,
                                    seed_base=seed_for_pair, vec_cap=vec_cap,
                                    dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level,
                                    log_info=True, sign_label="+"
                                )

                            for tid in tasks_used:
                                t = per_task_tensors[tid]
                                ro_plus = ros_plus_by_tid.get(tid, t["ro"])

                                if use_pg_inner:
                                    bc_phi = meta_objective_with_inner_pg(
                                        policy_net,
                                        ro_plus,
                                        t["batch_obs"], t["batch_lab"],
                                        args, device,
                                        alpha=getattr(args, "es_inner_pg_alpha", 0.0),
                                        scope=getattr(args, "es_inner_pg_scope", "head"),
                                        use_is=inner_use_is,
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(getattr(args, "es_ess_min_ratio", None) if not getattr(args, "es_recollect_inner", False) else None),
                                    )
                                else:
                                    bc_phi = meta_objective_from_rollout(
                                        policy_net,
                                        ro_plus,
                                        t["batch_obs"], t["batch_lab"],
                                        args, device,
                                        use_is_inner=(not getattr(args, "es_recollect_inner", False)) and bool(getattr(args, "es_use_is_inner", False)),
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(getattr(args, "es_ess_min_ratio", None) if not getattr(args, "es_recollect_inner", False) else None),
                                    )
                                bc_list_plus.append(bc_phi)

                            # drop tasks with NaN (ESS collapse)
                            bc_list_plus = [x for x in bc_list_plus if torch.isfinite(x)]
                            if bc_list_plus:
                                f_plus = torch.stack(bc_list_plus).mean()
                                if bool(getattr(args, "es_ranknorm", False)):
                                    vals = torch.stack(bc_list_plus)
                                    ranks = torch.argsort(torch.argsort(vals))
                                    f_plus = ((ranks.float() + 0.5) / float(len(vals))).mean()
                            else:
                                f_plus = torch.tensor(f_theta_baseline, device=device)

                        # -------- f(θ - σ ε)
                        with PerturbContext(named_params, eps, getattr(args, "es_sigma", 0.02), -1):
                            bc_list_minus = []
                            ros_minus_by_tid: Dict[int, ExploreRollout] = {}
                            if getattr(args, "es_recollect_inner", False):
                                seed_for_pair = (pair_seed if pair_seed is not None else args.seed + 99991 + i)
                                # Force INFO logs for trajectory collection
                                ros_minus_by_tid = _recollect_batch_for_sign(
                                    policy_net, per_task_tensors, tasks_used, device,
                                    seed_base=seed_for_pair, vec_cap=vec_cap,
                                    dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level,
                                    log_info=True, sign_label="-"
                                )

                            for tid in tasks_used:
                                t = per_task_tensors[tid]
                                ro_minus = ros_minus_by_tid.get(tid, t["ro"])

                                if use_pg_inner:
                                    bc_phi = meta_objective_with_inner_pg(
                                        policy_net,
                                        ro_minus,
                                        t["batch_obs"], t["batch_lab"],
                                        args, device,
                                        alpha=getattr(args, "es_inner_pg_alpha", 0.0),
                                        scope=getattr(args, "es_inner_pg_scope", "head"),
                                        use_is=inner_use_is,
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(getattr(args, "es_ess_min_ratio", None) if not getattr(args, "es_recollect_inner", False) else None),
                                    )
                                else:
                                    bc_phi = meta_objective_from_rollout(
                                        policy_net,
                                        ro_minus,
                                        t["batch_obs"], t["batch_lab"],
                                        args, device,
                                        use_is_inner=(not getattr(args, "es_recollect_inner", False)) and bool(getattr(args, "es_use_is_inner", False)),
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(getattr(args, "es_ess_min_ratio", None) if not getattr(args, "es_recollect_inner", False) else None),
                                    )
                                bc_list_minus.append(bc_phi)

                            bc_list_minus = [x for x in bc_list_minus if torch.isfinite(x)]
                            if bc_list_minus:
                                f_minus = torch.stack(bc_list_minus).mean()
                                if bool(getattr(args, "es_ranknorm", False)):
                                    vals = torch.stack(bc_list_minus)
                                    ranks = torch.argsort(torch.argsort(vals))
                                    f_minus = ((ranks.float() + 0.5) / float(len(vals))).mean()
                            else:
                                f_minus = torch.tensor(f_theta_baseline, device=device)

                        # ES gradient estimate: g += [(f+ - f-) / (2σ)] * ε
                        if bool(getattr(args, "es_fitness_baseline", False)):
                            coeff = ((f_plus - f_theta_baseline) - (f_minus - f_theta_baseline)) / (2.0 * float(getattr(args, "es_sigma", 0.02)))
                        else:
                            coeff = (f_plus - f_minus) / (2.0 * float(getattr(args, "es_sigma", 0.02)))
                        for n, p in named_params.items():
                            es_grads[n] = es_grads[n].add_(coeff * eps[n])

                    # Average over population
                    for n in es_grads:
                        es_grads[n].div_(float(getattr(args, "es_popsize", 8)))

                    # Apply gradients via existing optimizer
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            p.grad = None
                    for n, p in named_params.items():
                        p.grad = es_grads[n]
                    if getattr(args, "es_clip_grad", 0.0) and getattr(args, "es_clip_grad", 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(list(named_params.values()), getattr(args, "es_clip_grad", 1.0))
                    optimizer.step()

                    # housekeeping
                    for tid in tasks_used:
                        explore_cache[tid].reuse_count += 1
                    if device.type == "cuda":
                        updates_since_free += 1
                        if updates_since_free >= free_every:
                            torch.cuda.empty_cache()
                            updates_since_free = 0

                    if log_every_step or is_task_verbose:
                        msg = f"[ES STEP {step_idx+1}/{args.nbc}] bc(avg)={running_bc/max(1,count_updates):.4f}"
                        if debug_mem:
                            msg += " | " + _cuda_mem("post-es", device)
                        logger.info(msg)

                    continue  # ES branch finished this step

                # ----------- Backprop branch (original, pure-BC outer) -----------
                optimizer.zero_grad(set_to_none=True)

                loss_bc_list = []
                delta_list = []
                inrl_monitor_sum = 0.0

                for idx_task, tid in enumerate(tasks_used):
                    tensors = per_task_tensors[tid]
                    exp = tensors["exp"]; acts = tensors["actions"]; beh = tensors["beh_logits"]
                    rews_raw = tensors["rewards"]
                    rews = torch.clamp(rews_raw * args.rew_scale, -args.rew_clip, args.rew_clip)

                    # (optional) truncate explore window for speed
                    if args.adapt_trunc_T is not None and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
                        exp = exp[-args.adapt_trunc_T:]; acts = acts[-args.adapt_trunc_T:]; rews = rews[-args.adapt_trunc_T:]; beh  = beh[-args.adapt_trunc_T:]
                    if is_task_verbose and debug_shapes and idx_task < debug_tasks_per_batch and step_idx == 0:
                        logger.debug("[PLASTIC][TRUNC] tid=%s Tx_used=%d", tid, exp.shape[0])

                    # ---- pass 0: BC at θ (no plastic) for Δ gate ----
                    with torch.no_grad():
                        if hasattr(policy_net, "reset_plastic"):
                            policy_net.reset_plastic(batch_size=1, device=device)
                            policy_net.set_plastic(update_traces=False, modulators=None)
                        logits_bc_theta, _ = policy_net(tensors["batch_obs"])
                        if args.label_smoothing > 0.0:
                            loss_bc_theta = smoothed_cross_entropy(
                                logits_bc_theta, tensors["batch_lab"], ignore_index=PAD_ACTION, smoothing=args.label_smoothing
                            )
                        else:
                            ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_theta = ce(
                                logits_bc_theta.reshape(-1, logits_bc_theta.size(-1)),
                                tensors["batch_lab"].reshape(-1)
                            )

                    # ---- pass 1: advantages at θ + modulators ----
                    if hasattr(policy_net, "reset_plastic"):
                        policy_net.reset_plastic(batch_size=1, device=device)
                        policy_net.set_plastic(update_traces=False, modulators=None)
                    logits_theta, _ = policy_net(exp.unsqueeze(0))  # (1,T,A) with grad
                    logits_theta = logits_theta[0]

                    returns = discounted_returns(rews, args.gamma)
                    # per-task linear baseline
                    T = returns.size(0)
                    t = torch.arange(T, device=returns.device, dtype=returns.dtype)
                    zt = (t - t.mean()) / t.std().clamp_min(1e-6)
                    F = torch.stack([torch.ones_like(zt), zt], dim=1)  # (T,2)
                    FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
                    w = torch.linalg.solve(FT_F, F.T @ returns)
                    baseline = (F @ w)

                    adv = returns - baseline
                    adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
                    m_t = adv_norm.clamp(-args.plastic_clip_mod, args.plastic_clip_mod)

                    # Optional: IS-corrected plasticity when reusing behavior data
                    if getattr(args, "inner_use_is_mod", False):
                        lp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                        lp_beh = torch.log_softmax(beh,           dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                        rho = torch.exp(lp_cur - lp_beh)
                        if args.is_clip_rho and args.is_clip_rho > 0:
                            rho = torch.clamp(rho, max=args.is_clip_rho)
                        m_t = m_t * rho.detach()

                    if is_task_verbose and debug_inner_per_task and idx_task < debug_tasks_per_batch:
                        logger.debug("[PLASTIC][MOD] tid=%s mean=%.3f std=%.3f min=%.3f max=%.3f",
                                     tid, float(m_t.mean()), float(m_t.std()), float(m_t.min()), float(m_t.max()))

                    # ---- pass 2: adapt traces with modulators, then BC at φ ----
                    if hasattr(policy_net, "reset_plastic"):
                        policy_net.reset_plastic(batch_size=1, device=device)
                        policy_net.set_plastic(update_traces=True, modulators=m_t.unsqueeze(0))
                    _ = policy_net(exp.unsqueeze(0))  # update fast weights

                    policy_net.set_plastic(update_traces=False, modulators=None)
                    with autocast(device_type="cuda", enabled=use_amp):
                        logits_phi, _ = policy_net(tensors["batch_obs"])
                        if args.label_smoothing > 0.0:
                            loss_bc_phi = smoothed_cross_entropy(logits_phi, tensors["batch_lab"], ignore_index=PAD_ACTION, smoothing=args.label_smoothing)
                        else:
                            ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_phi = ce(logits_phi.reshape(-1, logits_phi.size(-1)), tensors["batch_lab"].reshape(-1))
                    loss_bc_list.append(loss_bc_phi)

                    # Δ gate: positive => adaptation helped demos (for logging/optional weighting)
                    delta_k = (loss_bc_theta.detach() - loss_bc_phi.detach())
                    delta_list.append(delta_k)

                    # Monitor-only PG at θ (no IS weight here for monitoring stat)
                    with torch.no_grad():
                        logp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                        monitor_pg = - (logp_cur * adv_norm).mean()
                        inrl_monitor_sum += float(monitor_pg.detach().cpu())

                if len(loss_bc_list) == 0:
                    continue

                # --- Keep outer loss pure BC; optionally reweight by standardized Δ across tasks ---
                total_bc = sum(loss_bc_list) / float(len(loss_bc_list))
                if getattr(args, "delta_weight_mode", "none") != "none":
                    deltas = torch.stack(delta_list)
                    dnorm = (deltas - deltas.mean()) / deltas.std().clamp_min(1e-6)
                    if getattr(args, "delta_weight_mode", "none") == "relu":
                        dnorm = torch.relu(dnorm)
                    # turn standardized Δ into positive weights
                    w = (dnorm - dnorm.min()).detach() + 1e-6
                    w = w / w.sum().clamp_min(1e-6)
                    total_bc = (torch.stack(loss_bc_list) * w).sum()

                avg_loss = total_bc

                # Safe AMP step
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    logger.warning("[PLASTIC][NONFINITE] grad_norm=%s — skipping optimizer step", str(grad_norm))
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                for tid in tasks_used:
                    explore_cache[tid].reuse_count += 1
                running_bc += float(total_bc.detach().cpu())
                running_inrl += inrl_monitor_sum / max(1, len(tasks_used))
                count_updates += 1

                if device.type == "cuda":
                    updates_since_free += 1
                    if updates_since_free >= free_every:
                        torch.cuda.empty_cache()
                        updates_since_free = 0

                # Only spam step logs at DEBUG (unless --log_every_step is set)
                if log_every_step or is_task_verbose:
                    msg = f"[STEP {step_idx+1}/{args.nbc}] avg_bc={float(total_bc):.4f} inrl@theta(avg)={running_inrl/max(1,count_updates):.4f}"
                    if debug_mem:
                        msg += " | " + _cuda_mem("post-step", device)
                    logger.info(msg)

            per_task_tensors.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if count_updates > 0 and not getattr(args, "no_tqdm", False):
                pbar.set_postfix(
                    bc=f"{running_bc/max(1,count_updates):.3f}",
                    inrl=f"{running_inrl/max(1,count_updates):.3f}"
                )

        # ------- Validation -------
        policy_net.eval()
        with torch.no_grad():
            (val_results, avg_p1, avg_p2, std_p1, std_p2, avg_total, success_rate) = eval_sampled_val(
                policy_net, val_tasks, make_base_env(), device, sample_n=args.val_sample_size
            )

        logger.info(
            f"[Epoch {epoch:02d}] bc_outer={running_bc/max(1,count_updates):.4f} "
            f"inrl@theta={running_inrl/max(1,count_updates):.4f} "
            f"val_phase1={avg_p1:.2f}±{std_p1:.2f} val_phase2={avg_p2:.2f}±{std_p2:.2f} "
            f"success_rate={success_rate:.2f} avg_total_steps={avg_total:.2f}"
        )

        improved_this = False
        if avg_total < best_val_score:
            best_val_score, best_epoch = avg_total, epoch
            patience, improved_this = 0, True
            save_checkpoint(policy_net, epoch, best_val_score, args.save_path, args.load_path)
            logger.info("[CKPT] improved avg_total -> %.2f (epoch %d)", best_val_score, best_epoch)
        else:
            patience += (0 if getattr(args, "disable_early_stop", False) else 1)

        if getattr(args, "disable_early_stop", False) and not improved_this:
            save_checkpoint(policy_net, epoch, best_val_score, args.save_path, args.load_path)
            logger.info("[CKPT] saved (no early-stop) best=%.2f epoch=%d", best_val_score, best_epoch)

        if (not getattr(args, "disable_early_stop", False)) and patience >= args.early_stop_patience and not improved_this:
            logger.info(f"[EARLY STOP] no improvement for {patience} epochs, stopping.")
            break

    logger.info("BC meta training complete.")

if __name__ == "__main__":
    run_training()
