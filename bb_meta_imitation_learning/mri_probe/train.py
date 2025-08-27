# mri_train/train.py

import os, sys, json, math, random, logging, gc
from typing import Dict, List, Optional, Tuple, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

# ---- TF32 (Ada/A40) ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from bb_meta_imitation_learning.env.maze_task import MazeTaskManager

from .config import parse_args
from .model import build_model, save_checkpoint, load_checkpoint
from .model import find_encoder_module as _find_encoder_module
from .model import critic_param_names as _critic_param_names
from .utils import (
    make_task_id,
    load_all_manifests,
    eval_sampled_val,
    SEQ_LEN,
    PAD_ACTION,
    smoothed_cross_entropy,
)
from .utils import (
    _setup_logging,
    _cuda_mem,
    _shape_str,
    _count_params,
    _stats_from_list,
    _fmt_stats,
    _fmt_ids,
)
from .data import (
    concat_explore_and_exploit,
    assert_start_goal_match,
    load_phase2_six_and_labels,
    maybe_augment_demo_six_cpu,
    select_demo_paths_for_task as _select_demo_paths_for_task,
)
from .explore import ExploreRollout, collect_explore_vec, make_base_env
from .rl_loss import mean_kl_logits, ess_ratio_from_rhos, discounted_returns

logger = logging.getLogger(__name__)

# ---- functional_call compatibility (PyTorch 2.0+ uses torch.func) ----
try:
    from torch.func import functional_call as _functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call as _functional_call


# ---------- helpers ----------

def _pad_and_stack_cpu(obs_list: List[torch.Tensor], lab_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad to common T on CPU and stack -> (B, T, 6,H,W), (B, T)."""
    B = len(obs_list)
    T_max = max(x.shape[0] for x in obs_list)
    H, W = obs_list[0].shape[-2], obs_list[0].shape[-1]
    pad_obs, pad_lab = [], []
    for x, y in zip(obs_list, lab_list):
        t = x.shape[0]
        if t < T_max:
            pad_t = T_max - t
            x = torch.cat([torch.zeros((pad_t, 6, H, W), dtype=x.dtype), x], dim=0)
            y = torch.cat([torch.full((pad_t,), PAD_ACTION, dtype=y.dtype), y], dim=0)
        pad_obs.append(x)
        pad_lab.append(y)
    return torch.stack(pad_obs, dim=0), torch.stack(pad_lab, dim=0)

def _named_trainable(module: nn.Module) -> "OrderedDict[str, torch.nn.Parameter]":
    from collections import OrderedDict
    return OrderedDict((n, p) for n, p in module.named_parameters() if p.requires_grad)

def _functional_bc_loss_at_phi(model: nn.Module,
                               params_phi: Dict[str, torch.Tensor],
                               batch_obs: torch.Tensor,
                               batch_lab: torch.Tensor,
                               smoothing: float,
                               use_amp: bool):
    """
    Stateless forward at adapted parameters φ to compute BC loss.
    """
    try:
        with autocast(device_type="cuda", enabled=use_amp):
            logits_phi, _ = _functional_call(model, params_phi, (batch_obs,))
            if smoothing > 0.0:
                loss_bc_phi = smoothed_cross_entropy(logits_phi, batch_lab, ignore_index=PAD_ACTION, smoothing=smoothing)
            else:
                ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                loss_bc_phi = ce(logits_phi.reshape(-1, logits_phi.size(-1)), batch_lab.reshape(-1))
        return loss_bc_phi
    except Exception:
        # Fallback: load params in-place (slow, but robust), then restore.
        sd_backup = model.state_dict()
        try:
            sd_phi = {**sd_backup}
            for n in params_phi:
                sd_phi[n] = params_phi[n]
            model.load_state_dict(sd_phi, strict=False)
            with autocast(device_type="cuda", enabled=use_amp):
                logits_phi, _ = model(batch_obs)
                if smoothing > 0.0:
                    loss_bc_phi = smoothed_cross_entropy(logits_phi, batch_lab, ignore_index=PAD_ACTION, smoothing=smoothing)
                else:
                    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                    loss_bc_phi = ce(logits_phi.reshape(-1, logits_phi.size(-1)), batch_lab.reshape(-1))
            return loss_bc_phi
        finally:
            model.load_state_dict(sd_backup, strict=False)

def _rebuild_task_entry_from_ro(entry: Dict[str, torch.Tensor],
                                ro: ExploreRollout,
                                device: torch.device,
                                args,
                                npz_cache: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Rebuild explore+exploit tensors for a task entry after a fresh recollect."""
    exp_six_cpu = ro.obs6
    actions_cpu = ro.actions
    rewards_cpu = ro.rewards
    beh_logits_cpu = ro.beh_logits
    Tx = exp_six_cpu.shape[0]
    prev_action_start = float(actions_cpu[-1].item()) if Tx > 0 else 0.0

    demo_obs_list: List[torch.Tensor] = []
    demo_lab_list: List[torch.Tensor] = []
    for demo_path in entry["selected_paths"]:
        pre = npz_cache.pop(demo_path, None)
        p2_six, p2_labels = load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start, preloaded=pre)
        if p2_six.numel() == 0:
            continue
        p2_six = maybe_augment_demo_six_cpu(p2_six, args)
        obs6_cat, labels_cat = concat_explore_and_exploit(exp_six_cpu, p2_six, p2_labels)
        demo_obs_list.append(obs6_cat)
        demo_lab_list.append(labels_cat)

    if len(demo_obs_list) == 0:
        # Nothing to supervise; keep explore cache but don't touch tensors.
        entry["ro"] = ro
        return entry

    batch_obs_cpu, batch_lab_cpu = _pad_and_stack_cpu(demo_obs_list, demo_lab_list)
    if getattr(args, "pin_memory", False):
        # Only pin the short-lived batch tensors
        batch_obs_cpu = batch_obs_cpu.pin_memory()
        batch_lab_cpu = batch_lab_cpu.pin_memory()

    entry.update({
        "batch_obs_cpu": batch_obs_cpu,
        "batch_lab_cpu": batch_lab_cpu,
        "exp_cpu": exp_six_cpu,
        "actions_cpu": actions_cpu,
        "rewards_cpu": rewards_cpu,
        "beh_logits_cpu": beh_logits_cpu,
        "ro": ro,
    })

    # Refresh device copies immediately (no need to wait for the next copy-stream)
    entry["batch_obs_dev"]   = batch_obs_cpu.to(device, non_blocking=True)
    entry["batch_lab_dev"]   = batch_lab_cpu.to(device, non_blocking=True)
    entry["exp_dev"]         = exp_six_cpu.to(device, non_blocking=True)
    entry["actions_dev"]     = actions_cpu.to(device, non_blocking=True)
    entry["rewards_dev"]     = rewards_cpu.to(device, non_blocking=True)
    entry["beh_logits_dev"]  = beh_logits_cpu.to(device, non_blocking=True)
    return entry


# ---------- main ----------

def run_training():
    # Better fragmentation behavior (ideally set before process start)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from .remap import remap_pretrained_state
    args = parse_args()

    # Logging / threads
    chosen_level = (args.log_level or args.debug_level or ("DEBUG" if args.debug else "INFO")).upper()
    _setup_logging(log_file=args.log_file, level=chosen_level)
    torch.set_num_threads(getattr(args, "cpu_threads", min(16, os.cpu_count() or 8)))
    os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
    if getattr(args, "cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # Seeds
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # may be overridden by cudnn.benchmark
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

    # Load demo manifests: main + optional synthetic
    demos_by_task_main = load_all_manifests(args.demo_root)
    if not demos_by_task_main:
        raise RuntimeError("No demo manifests found in demo_root.")
    logger.info("[DATA] main demos: root=%s tasks=%d", args.demo_root, len(demos_by_task_main))

    demos_by_task_syn = {}
    if getattr(args, "syn_demo_root", None):
        try:
            demos_by_task_syn = load_all_manifests(args.syn_demo_root)
            logger.info("[DATA] syn demos: root=%s tasks=%d", args.syn_demo_root, len(demos_by_task_syn))
        except Exception as e:
            logger.warning("[DATA] syn_demo_root=%s could not be loaded (%s); continuing without synthetic demos.",
                           str(args.syn_demo_root), repr(e))

    # Merge tasks with sanity checks
    tasks = []
    all_tids = set(demos_by_task_main.keys()) | set(demos_by_task_syn.keys())
    for tid in all_tids:
        recs_main = demos_by_task_main.get(tid, [])
        recs_syn  = demos_by_task_syn.get(tid, [])
        if not recs_main and not recs_syn:
            continue
        if tid in task_index_to_dict:
            tdict = task_index_to_dict[tid]
        elif tid in task_hash_to_dict:
            tdict = task_hash_to_dict[tid]
        else:
            continue
        if recs_main: assert_start_goal_match(recs_main, tdict, tid)
        if recs_syn:  assert_start_goal_match(recs_syn,  tdict, tid)
        tasks.append({"task_id": tid, "task_dict": tdict, "recs_main": recs_main, "recs_syn": recs_syn})
    if not tasks:
        raise RuntimeError("No tasks with demos found across main/synthetic roots.")

    random.shuffle(tasks)
    n_val = min(len(tasks), args.val_size)
    val_tasks = tasks[:n_val]
    train_tasks = tasks[n_val:]
    logger.info("[SPLIT] train=%d val=%d (val_size=%d)", len(train_tasks), len(val_tasks), n_val)

    # Model
    policy_net = build_model(seq_len=SEQ_LEN).to(device)
    tot, trn = _count_params(policy_net)
    logger.info("[MODEL] built model params total=%d trainable=%d", tot, trn)

    # Init from BC if provided
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
    else:
        logger.info("[INIT] No BC init provided.")

    # Optimizer (encoder two-group schedule if present)
    encoder_module, enc_name = _find_encoder_module(policy_net)
    critic_names = _critic_param_names(policy_net)
    all_named_params = dict(policy_net.named_parameters())
    critic_params = [all_named_params[n] for n in critic_names]
    non_critic_params = [p for n, p in all_named_params.items() if n not in critic_names]

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
        logger.info("[CRITIC] No value-head params detected; critic aux updates will be skipped.")
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

    # EMA baseline for correction (if enabled)
    ema_baseline = None

    # CPU pool & small next-batch cache
    cpu_pool = ThreadPoolExecutor(max_workers=int(getattr(args, "cpu_workers", max(4, (os.cpu_count() or 8) // 2))))
    npz_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    patience = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info("========== [EPOCH %02d/%02d] ==========", epoch, args.epochs)

        if have_encoder and warmup_epochs > 0 and epoch == (warmup_epochs + 1):
            for p in encoder_module.parameters(): p.requires_grad = True
            enc_params = list(encoder_module.parameters())
            enc_ids = {id(p) for p in enc_params}
            rest_params = [p for n, p in all_named_params.items() if (id(p) not in enc_ids) and (n not in critic_names)]
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
        running_pg = 0.0
        running_corr = 0.0
        count_updates = 0

        pbar = tqdm(range(num_batches), desc=f"[Epoch {epoch:02d}] train", leave=False, disable=getattr(args, "no_tqdm", False))
        for b in pbar:
            start = b * args.batch_size
            end = min(len(train_tasks), (b + 1) * args.batch_size)
            batch_ids = [batch_indices[i] for i in range(start, end)]
            batch_tasks = [train_tasks[i] for i in batch_ids]
            task_by_id = {t["task_id"]: t for t in batch_tasks}

            # next-batch NPZ prefetch (CPU)
            if int(getattr(args, "prefetch_batches", 1)) > 0 and (b + 1) < num_batches:
                nstart = (b + 1) * args.batch_size
                nend = min(len(train_tasks), (b + 2) * args.batch_size)
                next_ids = [batch_indices[i] for i in range(nstart, nend)]
                next_tasks = [train_tasks[i] for i in next_ids]
                prefetch_paths: List[str] = []
                for nt in next_tasks:
                    sel = _select_demo_paths_for_task(nt.get("recs_main", []), nt.get("recs_syn", []), args, epoch=epoch)
                    prefetch_paths.extend(sel)
                futs = []
                for pth in prefetch_paths:
                    if pth in npz_cache:
                        continue
                    futs.append(cpu_pool.submit(_safe_npz_load_obs_acts, pth))
                for fut in futs:
                    try:
                        pth, obs, acts = fut.result()
                        if (obs is not None) and (acts is not None):
                            npz_cache[pth] = (obs, acts)
                    except Exception as e:
                        logger.warning("[PREFETCH][NPZ] failed: %s", repr(e))

            # collect explores where needed (base recollect for new/overused)
            need_collect = []
            for task in batch_tasks:
                tid = task["task_id"]
                if tid not in explore_cache or getattr(explore_cache[tid], "reuse_count", 0) >= args.explore_reuse_M:
                    need_collect.append(task)

            _vec_cap = int(getattr(args, "num_envs", 8))
            if need_collect:
                logger.info("[COLLECT][BASE] epoch=%d batch=%d vec_cap=%d seed_base=%d tasks=%s",
                            epoch, b + 1, _vec_cap,
                            args.seed + 100000 * epoch + 1000 * b,
                            _fmt_ids([t["task_id"] for t in need_collect]))
            for off in range(0, len(need_collect), max(1, _vec_cap)):
                slice_tasks = need_collect[off: off + max(1, _vec_cap)]
                cfgs = [MazeTaskManager.TaskConfig(**t["task_dict"]) for t in slice_tasks]
                seed_here = args.seed + 100000 * epoch + 1000 * b + off
                ro_list = collect_explore_vec(policy_net, cfgs, device, max_steps=250,
                                              seed_base=seed_here, dbg=False)
                for ttask, ro in zip(slice_tasks, ro_list):
                    explore_cache[ttask["task_id"]] = ro
                    try:
                        ro.reuse_count = 0
                    except Exception:
                        pass

            # ============== TWO-PASS BATCH PREP ==============
            # PASS 1 — compute KL/ESS vs current policy (stacked forward) & decide early refresh
            kls, esss, rmu, rsd, rmx = [], [], [], [], []
            stale_by_kl: List[int] = []
            stale_by_ess: List[int] = []

            # Build pad-batched Episode-1 tensors for current batch
            exps, behs, acts, lens, tids_order = [], [], [], [], []
            for task in batch_tasks:
                tid = task["task_id"]
                ro = explore_cache.get(tid)
                if ro is None:
                    cfg = MazeTaskManager.TaskConfig(**task["task_dict"])
                    ro = collect_explore_vec(policy_net, [cfg], device, max_steps=250,
                                             seed_base=args.seed + 424242, dbg=False).pop()
                    explore_cache[tid] = ro
                exps.append(ro.obs6)
                behs.append(ro.beh_logits)
                acts.append(ro.actions)
                lens.append(int(ro.obs6.shape[0]))
                tids_order.append(tid)

            if len(exps) == 0:
                continue

            B = len(exps)
            T_max = max(lens)
            C, H, W = exps[0].shape[1:]
            A = behs[0].shape[1]
            # Pad to (B,T_max,...) on device for a single forward
            exp_b = torch.zeros(B, T_max, C, H, W, device=device)
            beh_b = torch.zeros(B, T_max, A, device=device)
            act_b = torch.full((B, T_max), -1, dtype=torch.long, device=device)
            for i in range(B):
                tlen = lens[i]
                if tlen == 0: continue
                exp_b[i, :tlen] = exps[i].to(device, non_blocking=True)
                beh_b[i, :tlen] = behs[i].to(device, non_blocking=True)
                act_b[i, :tlen] = acts[i].to(device, non_blocking=True)

            with torch.no_grad(), autocast(device_type="cuda", enabled=use_amp):
                logits_now_b, _ = policy_net(exp_b)  # (B, T_max, A)

            # Per-task metrics
            for i in range(B):
                tlen = lens[i]
                if tlen == 0:
                    kls.append(0.0); esss.append(1.0); rmu.append(1.0); rsd.append(0.0); rmx.append(1.0)
                    continue
                li = logits_now_b[i, :tlen]
                bi = beh_b[i, :tlen]
                ai = act_b[i, :tlen]
                lp_now = torch.log_softmax(li, dim=-1).gather(1, ai.unsqueeze(1)).squeeze(1)
                lp_beh = torch.log_softmax(bi, dim=-1).gather(1, ai.unsqueeze(1)).squeeze(1)
                rhos = torch.exp(lp_now - lp_beh)
                if getattr(args, "is_clip_rho", 0.0) > 0:
                    rhos = torch.clamp(rhos, max=args.is_clip_rho)
                ess_ratio_val = ess_ratio_from_rhos(rhos).item()
                kl_val = mean_kl_logits(li, bi).item()

                kls.append(kl_val); esss.append(ess_ratio_val)
                rmu.append(float(rhos.mean().item()))
                rsd.append(float(rhos.std(unbiased=False).item()))
                rmx.append(float(rhos.max().item()))

                tid = tids_order[i]
                if kl_val > args.kl_refresh_threshold:
                    stale_by_kl.append(tid)
                if args.ess_refresh_ratio and args.ess_refresh_ratio > 0:
                    if int(getattr(args, "explore_reuse_M", 1)) > 1 and ess_ratio_val < args.ess_refresh_ratio:
                        stale_by_ess.append(tid)

            # Vectorized early refresh (union of stale)
            kl_refresh_count = len(stale_by_kl)
            ess_refresh_count = len(stale_by_ess)
            early_stale_tids = sorted(set(stale_by_kl) | set(stale_by_ess))
            if early_stale_tids:
                logger.info("[COLLECT][EARLY-REFRESH] epoch=%d batch=%d stale=%d (KL>%g: %d, ESS<%g: %d)",
                            epoch, b + 1, len(early_stale_tids),
                            float(args.kl_refresh_threshold), kl_refresh_count,
                            float(args.ess_refresh_ratio), ess_refresh_count)
                vec_cap = int(getattr(args, "num_envs", 8))
                for off in range(0, len(early_stale_tids), max(1, vec_cap)):
                    tids_slice = early_stale_tids[off: off + max(1, vec_cap)]
                    cfgs = [MazeTaskManager.TaskConfig(**task_by_id[tid]["task_dict"]) for tid in tids_slice]
                    seed_here = args.seed + 800000 + 1000 * b + off
                    ro_list = collect_explore_vec(policy_net, cfgs, device, max_steps=250,
                                                  seed_base=seed_here, dbg=False)
                    for tid, ro_new in zip(tids_slice, ro_list):
                        explore_cache[tid] = ro_new
                        try:
                            ro_new.reuse_count = 0
                        except Exception:
                            pass

            # free forward tensors
            del exp_b, beh_b, act_b, logits_now_b
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # PASS 2 — build per-task BC supervision tensors (using possibly refreshed rollouts)
            per_task_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
            tasks_used: List[int] = []

            for task in batch_tasks:
                tid = task["task_id"]
                ro = explore_cache.get(tid)
                if ro is None:
                    cfg = MazeTaskManager.TaskConfig(**task["task_dict"])
                    ro = collect_explore_vec(policy_net, [cfg], device, max_steps=250,
                                             seed_base=args.seed + 777777, dbg=False).pop()
                    explore_cache[tid] = ro

                exp_six_cpu = ro.obs6
                actions_cpu = ro.actions
                rewards_cpu = ro.rewards
                beh_logits_cpu = ro.beh_logits
                Tx = exp_six_cpu.shape[0]

                # Build batched BC tensors from selected demos (CPU -> pad)
                prev_action_start = float(actions_cpu[-1].item()) if Tx > 0 else 0.0
                selected_paths = _select_demo_paths_for_task(task.get("recs_main", []), task.get("recs_syn", []), args, epoch=epoch)

                demo_obs_list: List[torch.Tensor] = []
                demo_lab_list: List[torch.Tensor] = []
                for demo_path in selected_paths:
                    pre = npz_cache.pop(demo_path, None)
                    p2_six, p2_labels = load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start, preloaded=pre)
                    if p2_six.numel() == 0:
                        continue
                    p2_six = maybe_augment_demo_six_cpu(p2_six, args)
                    obs6_cat, labels_cat = concat_explore_and_exploit(exp_six_cpu, p2_six, p2_labels)
                    demo_obs_list.append(obs6_cat)
                    demo_lab_list.append(labels_cat)

                if len(demo_obs_list) == 0:
                    logger.warning("[BC][SKIP] tid=%s no demos to supervise in this batch", tid)
                    continue

                batch_obs_cpu, batch_lab_cpu = _pad_and_stack_cpu(demo_obs_list, demo_lab_list)
                if getattr(args, "pin_memory", False):
                    # Only pin short-lived batch tensors
                    batch_obs_cpu = batch_obs_cpu.pin_memory()
                    batch_lab_cpu = batch_lab_cpu.pin_memory()

                per_task_tensors[tid] = {
                    "batch_obs_cpu": batch_obs_cpu,
                    "batch_lab_cpu": batch_lab_cpu,
                    "exp_cpu": exp_six_cpu,
                    "actions_cpu": actions_cpu,
                    "rewards_cpu": rewards_cpu,
                    "beh_logits_cpu": beh_logits_cpu,
                    "ro": ro,
                    "task_dict": task["task_dict"],
                    "selected_paths": selected_paths,
                }
                tasks_used.append(tid)

            if len(tasks_used) == 0:
                continue

            # H2D copies on a dedicated stream
            if device.type == "cuda":
                copy_stream = torch.cuda.Stream()
                with torch.cuda.stream(copy_stream):
                    for tid in tasks_used:
                        t = per_task_tensors[tid]
                        t["batch_obs_dev"] = t["batch_obs_cpu"].to(device, non_blocking=True)
                        t["batch_lab_dev"] = t["batch_lab_cpu"].to(device, non_blocking=True)
                        t["exp_dev"] = t["exp_cpu"].to(device, non_blocking=True)
                        t["actions_dev"] = t["actions_cpu"].to(device, non_blocking=True)
                        t["rewards_dev"] = t["rewards_cpu"].to(device, non_blocking=True)
                        t["beh_logits_dev"] = t["beh_logits_cpu"].to(device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(copy_stream)

            # Batch indicators (use stats from PASS 1)
            s_kl  = _stats_from_list(kls)
            s_ess = _stats_from_list(esss)
            s_rmu = _stats_from_list(rmu)
            s_rsd = _stats_from_list(rsd)
            s_rmx = _stats_from_list(rmx)
            logger.info(
                "[IND][BATCH %d/%d] tasks=%d | %s | ESS:%s | rho(mean):%s rho(std):%s rho(max):%s | refresh: KL=%d ESS=%d",
                b + 1, num_batches, len(tasks_used),
                _fmt_stats("KL", s_kl), _fmt_stats("", s_ess),
                _fmt_stats("", s_rmu), _fmt_stats("", s_rsd), _fmt_stats("", s_rmx),
                kl_refresh_count, ess_refresh_count,
            )

            # =========================
            # MRI OUTER LOOP (streamed backward)
            # =========================
            for step_idx in range(args.nbc):
                optimizer.zero_grad(set_to_none=True)

                # names for inner updates (exclude critic head)
                named_all = _named_trainable(policy_net)
                adapt_names = [n for n in named_all.keys() if n not in _critic_param_names(policy_net)]

                tasks_n = len(tasks_used)
                total_bc_val = 0.0
                total_corr_val = 0.0
                pg_vals = []

                # running baseline if 'batch' is requested
                batch_mean_bc = 0.0
                batch_seen = 0

                for tid in tasks_used:
                    t = per_task_tensors[tid]
                    exp = t["exp_dev"]; acts = t["actions_dev"]; beh = t["beh_logits_dev"]
                    rews_raw = t["rewards_dev"]
                    rews = torch.clamp(rews_raw * args.rew_scale, -args.rew_clip, args.rew_clip)

                    # truncate explore window if needed
                    if args.adapt_trunc_T is not None and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
                        exp = exp[-args.adapt_trunc_T:]; acts = acts[-args.adapt_trunc_T:]; rews = rews[-args.adapt_trunc_T:]; beh  = beh[-args.adapt_trunc_T:]

                    # ----- Inner policy gradient steps (on θ) -----
                    params_k = OrderedDict((n, p) for n, p in named_all.items())
                    alpha = float(getattr(args, "inner_pg_alpha", 0.1))

                    for k in range(int(getattr(args, "inner_steps", 1))):
                        with autocast(device_type="cuda", enabled=use_amp):
                            logits_seq, _ = _functional_call(policy_net, params_k, (exp.unsqueeze(0),))
                            logits_seq = logits_seq[0]

                        # compute advantages with linear baseline
                        returns = discounted_returns(rews, args.gamma)
                        T = returns.size(0)
                        tt = torch.arange(T, device=returns.device, dtype=returns.dtype)
                        zt = (tt - tt.mean()) / tt.std().clamp_min(1e-6)
                        F = torch.stack([torch.ones_like(zt), zt], dim=1)
                        FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
                        w = torch.linalg.solve(FT_F, F.T @ returns)
                        baseline = (F @ w)
                        adv = returns - baseline
                        adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

                        logp_cur = torch.log_softmax(logits_seq, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)

                        if bool(getattr(args, "inner_use_is", False)):
                            lp_beh = torch.log_softmax(beh, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                            rho = torch.exp(logp_cur.detach() - lp_beh)
                            if getattr(args, "is_clip_rho", 0.0) > 0:
                                rho = torch.clamp(rho, max=args.is_clip_rho)
                            inner_loss = - ((rho * logp_cur) * adv_norm).mean()
                        else:
                            inner_loss = - (logp_cur * adv_norm).mean()

                        pg_vals.append(inner_loss.detach().item())

                        grads = torch.autograd.grad(
                            inner_loss,
                            [params_k[n] for n in adapt_names],
                            create_graph=bool(getattr(args, "second_order", False)),
                            retain_graph=bool(getattr(args, "second_order", False)),
                            allow_unused=False
                        )
                        # θ_{k+1} = θ_k - α * g_k
                        for (n, _p), g in zip(((n, params_k[n]) for n in adapt_names), grads):
                            params_k[n] = params_k[n] - alpha * g

                        del logits_seq, returns, adv, adv_norm, logp_cur
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    # ----- Outer BC at φ (params_k) -----
                    loss_bc_phi = _functional_bc_loss_at_phi(
                        policy_net, params_k, t["batch_obs_dev"], t["batch_lab_dev"],
                        smoothing=float(getattr(args, "label_smoothing", 0.0)), use_amp=use_amp
                    )
                    bc_scalar_detached = float(loss_bc_phi.detach().item())
                    total_bc_val += bc_scalar_detached

                    # ----- Meta-descent correction term (score-function) -----
                    corr_loss = 0.0
                    if bool(getattr(args, "meta_corr", False)):
                        with autocast(device_type="cuda", enabled=use_amp):
                            logits_theta, _ = policy_net(exp.unsqueeze(0))   # θ at current (pre-update) params
                            logits_theta = logits_theta[0]
                        logp_theta = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)

                        # IS weights; multiplier detached where appropriate
                        if bool(getattr(args, "meta_corr_use_is", False)):
                            lp_beh = torch.log_softmax(beh, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                            w_imp = torch.exp(logp_theta.detach() - lp_beh)
                            if getattr(args, "is_clip_rho", 0.0) > 0:
                                w_imp = torch.clamp(w_imp, max=args.is_clip_rho)
                        else:
                            w_imp = torch.ones_like(logp_theta)

                        if bool(getattr(args, "meta_corr_center_logp", False)):
                            # detach the weighted mean so gradients don't flow through centering value
                            wmean = ((w_imp * logp_theta).sum() / (w_imp.sum() + 1e-8)).detach()
                            logp_theta = logp_theta - wmean

                        sum_logp = (w_imp.detach() * logp_theta).mean()
                        coeff = float(getattr(args, "meta_corr_coeff", 1.0))

                        # baseline selection in streaming mode
                        if str(getattr(args, "meta_corr_baseline", "batch")) == "ema":
                            bval = float(ema_baseline) if (ema_baseline is not None) else 0.0
                        else:  # "batch": running mean of seen tasks
                            bval = float(batch_mean_bc) if batch_seen > 0 else 0.0

                        corr_loss = - coeff * (loss_bc_phi.detach() - bval) * sum_logp
                        total_corr_val += float(corr_loss.detach().item())

                        del logits_theta, logp_theta, w_imp, sum_logp

                    # ---- Stream backward now: free each task graph immediately ----
                    task_loss = loss_bc_phi + (corr_loss if bool(getattr(args, "meta_corr", False)) else 0.0)
                    scaler.scale(task_loss / tasks_n).backward()

                    # update baselines after using bval
                    if str(getattr(args, "meta_corr_baseline", "batch")) == "ema":
                        if ema_baseline is None:
                            ema_baseline = bc_scalar_detached
                        else:
                            ema_beta = float(getattr(args, "meta_corr_ema_beta", 0.9))
                            ema_baseline = ema_beta * ema_baseline + (1.0 - ema_beta) * bc_scalar_detached
                    else:
                        batch_seen += 1
                        batch_mean_bc = ((batch_seen - 1) * batch_mean_bc + bc_scalar_detached) / max(1, batch_seen)

                    # Help GC
                    del params_k, loss_bc_phi

                # one optimizer step for the whole step_idx
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer); scaler.update()
                else:
                    logger.warning("[MRI][NONFINITE] grad_norm=%s — skipping optimizer step", str(grad_norm))
                    optimizer.zero_grad(set_to_none=True); scaler.update()

                # bookkeeping
                for tid in tasks_used:
                    try:
                        explore_cache[tid].reuse_count = getattr(explore_cache[tid], "reuse_count", 0) + 1
                    except Exception:
                        pass

                running_bc += total_bc_val / max(1, tasks_n)
                running_pg += (np.mean(pg_vals) if pg_vals else 0.0)
                running_corr += (total_corr_val / max(1, tasks_n) if bool(getattr(args, "meta_corr", False)) else 0.0)
                count_updates += 1

                logger.info(
                    "[MRI][STEP %d/%d] outer_BC=%.4f | inner_pg(mean)=%.4f | corr_loss=%.4f | grad_norm=%.3e",
                    step_idx + 1, args.nbc,
                    float(total_bc_val / max(1, tasks_n)),
                    float(np.mean(pg_vals) if pg_vals else 0.0),
                    float(total_corr_val / max(1, tasks_n) if bool(getattr(args, "meta_corr", False)) else 0.0),
                    float(grad_norm if torch.is_tensor(grad_norm) else grad_norm),
                )

                # --- Mid-batch staleness probe (KL/ESS) & targeted recollect (VECTORIZED) ---
                probe_every = int(getattr(args, "stale_probe_every", 0) or 0)
                probe_K     = int(getattr(args, "stale_probe_K", 0) or 0)
                if probe_every > 0 and ((step_idx + 1) % probe_every == 0):
                    tids_to_check = list(tasks_used)

                    # Build selected (possibly truncated/subsampled) sequences per task
                    exps_sel, behs_sel, acts_sel, lens_sel, task_order = [], [], [], [], []
                    for tid in tids_to_check:
                        t = per_task_tensors[tid]
                        exp = t["exp_dev"]; beh = t["beh_logits_dev"]; act = t["actions_dev"]
                        # mirror inner truncation if configured
                        if args.adapt_trunc_T is not None and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
                            exp = exp[-args.adapt_trunc_T:]; beh = beh[-args.adapt_trunc_T:]; act = act[-args.adapt_trunc_T:]
                        Tlen = exp.shape[0]
                        if Tlen == 0:
                            continue
                        if probe_K > 0 and Tlen > probe_K:
                            idx = torch.randperm(Tlen, device=device)[:probe_K].sort()[0]
                            exp = exp.index_select(0, idx)
                            beh = beh.index_select(0, idx)
                            act = act.index_select(0, idx)
                            Tlen = exp.shape[0]
                        exps_sel.append(exp); behs_sel.append(beh); acts_sel.append(act); lens_sel.append(Tlen); task_order.append(tid)

                    if lens_sel:
                        Bp = len(exps_sel)
                        Tp = max(lens_sel)
                        C, H, W = exps_sel[0].shape[1:]
                        A = behs_sel[0].shape[1]
                        exp_pb = torch.zeros(Bp, Tp, C, H, W, device=device)
                        beh_pb = torch.zeros(Bp, Tp, A, device=device)
                        act_pb = torch.full((Bp, Tp), -1, dtype=torch.long, device=device)
                        for i in range(Bp):
                            tlen = lens_sel[i]
                            exp_pb[i, :tlen] = exps_sel[i]
                            beh_pb[i, :tlen] = behs_sel[i]
                            act_pb[i, :tlen] = acts_sel[i]

                        with torch.no_grad(), autocast(device_type="cuda", enabled=use_amp):
                            logits_now_b, _ = policy_net(exp_pb)  # (B,T,A)

                        # Per-task KL/ESS and decision
                        stale_tids: List[int] = []
                        bad_kl = 0; bad_ess = 0
                        for i, tid in enumerate(task_order):
                            tlen = lens_sel[i]
                            li = logits_now_b[i, :tlen]
                            bi = beh_pb[i, :tlen]
                            ai = act_pb[i, :tlen]

                            lp_now = torch.log_softmax(li, dim=-1).gather(1, ai.unsqueeze(1)).squeeze(1)
                            lp_beh = torch.log_softmax(bi, dim=-1).gather(1, ai.unsqueeze(1)).squeeze(1)
                            rhos   = torch.exp(lp_now - lp_beh)
                            if getattr(args, "is_clip_rho", 0.0) > 0:
                                rhos = torch.clamp(rhos, max=args.is_clip_rho)
                            ess_ratio_val = ess_ratio_from_rhos(rhos).item()
                            kl_val = mean_kl_logits(li, bi).item()

                            flag_kl  = (kl_val > args.kl_refresh_threshold)
                            flag_ess = (args.ess_refresh_ratio and args.ess_refresh_ratio > 0 and ess_ratio_val < args.ess_refresh_ratio)
                            if flag_kl or flag_ess:
                                stale_tids.append(tid)
                                bad_kl  += int(flag_kl)
                                bad_ess += int(flag_ess)

                        if stale_tids:
                            logger.info("[MIDPROBE][RECOLLECT] epoch=%d batch=%d step=%d probe_every=%d K=%d | stale=%d (KL>%g: %d, ESS<%g: %d)",
                                        epoch, b + 1, step_idx + 1, probe_every, probe_K,
                                        len(stale_tids), float(args.kl_refresh_threshold), bad_kl,
                                        float(args.ess_refresh_ratio), bad_ess)
                            vec_cap = int(getattr(args, "num_envs", 8))
                            for off in range(0, len(stale_tids), max(1, vec_cap)):
                                tids_slice = stale_tids[off : off + max(1, vec_cap)]
                                cfgs = [MazeTaskManager.TaskConfig(**per_task_tensors[tid]["task_dict"]) for tid in tids_slice]
                                seed_here = args.seed + 100000 * epoch + 1000 * b + (step_idx + 1) * 123 + off
                                ro_list = collect_explore_vec(policy_net, cfgs, device, max_steps=250, seed_base=seed_here, dbg=False)
                                for tid, ro_new in zip(tids_slice, ro_list):
                                    explore_cache[tid] = ro_new
                                    try:
                                        ro_new.reuse_count = 0
                                    except Exception:
                                        pass
                                    per_task_tensors[tid] = _rebuild_task_entry_from_ro(
                                        per_task_tensors[tid], ro_new, device, args, npz_cache
                                    )

                        # free batch probes
                        del exp_pb, beh_pb, act_pb, logits_now_b
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                # --- Mid-batch vectorized recollect to enforce the reuse budget ---
                reuse_budget = max(1, int(getattr(args, "explore_reuse_M", 1)))
                if (step_idx + 1) % reuse_budget == 0:
                    vec_cap = int(getattr(args, "num_envs", 8))
                    for off in range(0, len(tasks_used), max(1, vec_cap)):
                        tids_slice = tasks_used[off : off + max(1, vec_cap)]
                        cfgs = [MazeTaskManager.TaskConfig(**per_task_tensors[tid]["task_dict"]) for tid in tids_slice]
                        seed_here = args.seed + 100000 * epoch + 1000 * b + (step_idx + 1) + off
                        ro_list = collect_explore_vec(policy_net, cfgs, device, max_steps=250, seed_base=seed_here, dbg=False)
                        for tid, ro_new in zip(tids_slice, ro_list):
                            explore_cache[tid] = ro_new
                            try:
                                ro_new.reuse_count = 0
                            except Exception:
                                pass
                            # Rebuild this task's tensors against the fresh explore rollout
                            per_task_tensors[tid] = _rebuild_task_entry_from_ro(
                                per_task_tensors[tid], ro_new, device, args, npz_cache
                            )

                # Step-level cleanup
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # ----- End of batch cleanup -----
            per_task_tensors.clear()
            del per_task_tensors, tasks_used
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if count_updates > 0 and not getattr(args, "no_tqdm", False):
                pbar.set_postfix(
                    bc=f"{running_bc/max(1,count_updates):.3f}",
                    pg=f"{running_pg/max(1,count_updates):.3f}",
                    corr=f"{running_corr/max(1,count_updates):.3f}",
                )

        # ------- Validation -------
        policy_net.eval()
        with torch.no_grad():
            (val_results, avg_p1, avg_p2, std_p1, std_p2, avg_total, success_rate) = eval_sampled_val(
                policy_net, val_tasks, make_base_env(), device, sample_n=args.val_sample_size
            )

        logger.info(
            f"[Epoch {epoch:02d}] bc_outer={running_bc/max(1,count_updates):.4f} "
            f"inrl={running_pg/max(1,count_updates):.4f} corr={running_corr/max(1,count_updates):.4f} "
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

    cpu_pool.shutdown(wait=True)
    logger.info("BC meta training complete.")

def _safe_npz_load_obs_acts(path: str) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        d = np.load(path)
        return path, d["observations"], d["actions"]
    except Exception:
        return path, None, None

if __name__ == "__main__":
    run_training()
