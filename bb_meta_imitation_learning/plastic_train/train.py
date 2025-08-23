# plastic_train/train.py 

import os, sys, json, math, random, logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    _frac_clipped,
    _frac_at_clip_abs,
    _fmt_ids,
)
from .data import (
    concat_explore_and_exploit,
    first_demo_paths,
    assert_start_goal_match,
    load_phase2_six_and_labels,
    maybe_augment_demo_six_cpu,
)
from .data import select_demo_paths_for_task as _select_demo_paths_for_task
from .explore import ExploreRollout, collect_explore_vec, make_base_env
from .explore import recollect_batch_for_sign as _recollect_batch_for_sign
from .rl_loss import mean_kl_logits, ess_ratio_from_rhos, discounted_returns
from .rl_loss import compute_bc_at_theta as _compute_bc_theta
from .rl_loss import outer_pg_term as _outer_pg_term

# ---- ES helpers ----
from .es import (
    select_es_named_params,
    sample_eps,             # still available; we will pre-sample our own CPU eps bank
    PerturbContext,
    meta_objective_from_rollout,
    meta_objective_with_inner_pg,  # inner RL + post-adapt BC
)

logger = logging.getLogger(__name__)


def _sample_eps_bank_cpu(named_params: Dict[str, torch.Tensor], algo: str, popsize: int,
                         pin: bool = False) -> List[Dict[str, torch.Tensor]]:
    """
    Pre-sample ES/SPSA noise on CPU to avoid repeated GPU small allocations; optionally pin.
    """
    bank: List[Dict[str, torch.Tensor]] = []
    if algo == "spsa":
        for _ in range(popsize):
            eps = {n: torch.empty_like(p, device="cpu").bernoulli_(0.5).mul_(2).sub_(1) for n, p in named_params.items()}
            if pin:
                for k in eps:
                    eps[k] = eps[k].pin_memory()
            bank.append(eps)
    else:  # "es" -> Gaussian
        for _ in range(popsize):
            eps = {n: torch.randn(p.shape, dtype=p.dtype, device="cpu") for n, p in named_params.items()}
            if pin:
                for k in eps:
                    eps[k] = eps[k].pin_memory()
            bank.append(eps)
    return bank


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


def run_training():
    from .remap import remap_pretrained_state  # local import to keep module edges clean
    args = parse_args()

    # Prefer explicit --log_level; else --debug_level; else DEBUG if --debug else INFO.
    chosen_level = (args.log_level or args.debug_level or ("DEBUG" if args.debug else "INFO")).upper()
    _setup_logging(log_file=args.log_file, level=chosen_level)

    # CPU threads + OMP (helps packing/padding/npz decode)
    torch.set_num_threads(getattr(args, "cpu_threads", min(16, os.cpu_count() or 8)))
    os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
    if getattr(args, "cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

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
    torch.backends.cudnn.deterministic = True  # may be overridden by --cudnn_benchmark
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

    # Merge task entries
    tasks = []
    all_tids = set(demos_by_task_main.keys()) | set(demos_by_task_syn.keys())
    for tid in all_tids:
        recs_main = demos_by_task_main.get(tid, [])
        recs_syn = demos_by_task_syn.get(tid, [])
        if not recs_main and not recs_syn:
            continue
        # locate task dict by numeric index or hash
        if tid in task_index_to_dict:
            tdict = task_index_to_dict[tid]
        elif tid in task_hash_to_dict:
            tdict = task_hash_to_dict[tid]
        else:
            continue
        # sanity
        if recs_main:
            assert_start_goal_match(recs_main, tdict, tid)
        if recs_syn:
            assert_start_goal_match(recs_syn, tdict, tid)
        tasks.append({"task_id": tid, "task_dict": tdict, "recs_main": recs_main, "recs_syn": recs_syn})
    if not tasks:
        raise RuntimeError("No tasks with demos found across main/synthetic roots.")

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

    # ---------- Load BC init + remap ----------
    if args.bc_init:
        ck_path = os.path.abspath(args.bc_init)
        if not os.path.isfile(ck_path):
            raise FileNotFoundError(f"{ck_path} not found")
        from .remap import remap_pretrained_state
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
            for p in encoder_module.parameters():
                p.requires_grad = False
            optimizer = torch.optim.Adam(rest_params, lr=args.lr, weight_decay=args.weight_decay)
            logger.info(f"[ENCODER] Frozen for first {warmup_epochs} epoch(s).")
        else:
            for p in encoder_module.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                [
                    {"params": rest_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": enc_params, "lr": args.lr * args.encoder_lr_mult, "weight_decay": args.weight_decay},
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

    if debug_shapes:
        logger.info("[SHAPES] SEQ_LEN=%s", str(SEQ_LEN))

    # Thread pool for CPU work (decode / augment / padding)
    cpu_pool = ThreadPoolExecutor(max_workers=int(getattr(args, "cpu_workers", max(4, (os.cpu_count() or 8) // 2))))

    # Small CPU dict to hold look-ahead NPZs for the NEXT batch only (no cross-batch reuse beyond that)
    npz_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info("========== [EPOCH %02d/%02d] ==========", epoch, args.epochs)

        if have_encoder and warmup_epochs > 0 and epoch == (warmup_epochs + 1):
            for p in encoder_module.parameters():
                p.requires_grad = True
            enc_params = list(encoder_module.parameters())
            enc_ids = {id(p) for p in enc_params}
            rest_params = [p for n, p in all_named_params.items() if (id(p) not in enc_ids) and (n not in critic_names)]
            optimizer = torch.optim.Adam(
                [
                    {"params": rest_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": enc_params, "lr": args.lr * args.encoder_lr_mult, "weight_decay": args.weight_decay},
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
            is_debug_batch = debug and ((b % max(1, debug_every_batches)) == 0)
            is_task_verbose = is_debug_batch and (debug_level == "DEBUG")

            start = b * args.batch_size
            end = min(len(train_tasks), (b + 1) * args.batch_size)
            batch_ids = [batch_indices[i] for i in range(start, end)]
            batch_tasks = [train_tasks[i] for i in batch_ids]

            # Schedule CPU-only NPZ prefetch for NEXT batch (no model calls)
            if int(getattr(args, "prefetch_batches", 1)) > 0 and (b + 1) < num_batches:
                nstart = (b + 1) * args.batch_size
                nend = min(len(train_tasks), (b + 2) * args.batch_size)
                next_ids = [batch_indices[i] for i in range(nstart, nend)]
                next_tasks = [train_tasks[i] for i in next_ids]
                prefetch_paths: List[str] = []
                for nt in next_tasks:
                    sel = _select_demo_paths_for_task(nt.get("recs_main", []), nt.get("recs_syn", []), args, epoch=epoch)
                    prefetch_paths.extend(sel)
                # Launch NPZ loads on the CPU pool
                futs = []
                for pth in prefetch_paths:
                    if pth in npz_cache:
                        continue
                    futs.append(cpu_pool.submit(_safe_npz_load_obs_acts, pth))
                # Collect completed loads asynchronously; stash in npz_cache
                for fut in futs:
                    try:
                        pth, obs, acts = fut.result()
                        if (obs is not None) and (acts is not None):
                            npz_cache[pth] = (obs, acts)
                    except Exception as e:
                        logger.warning("[PREFETCH][NPZ] failed: %s", repr(e))

            if is_task_verbose:
                tids = [t["task_id"] for t in batch_tasks]
                logger.info("[BATCH %d/%d] tasks=%s", b + 1, num_batches, _fmt_ids(tids))
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
                logger.info(
                    "[COLLECT][BASE] epoch=%d batch=%d vec_cap=%d seed_base=%d tasks=%s",
                    epoch, b + 1, _vec_cap,
                    args.seed + 100000 * epoch + 1000 * b,
                    _fmt_ids([t["task_id"] for t in need_collect]),
                )

            for off in range(0, len(need_collect), max(1, _vec_cap)):
                slice_tasks = need_collect[off: off + max(1, _vec_cap)]
                cfgs = [MazeTaskManager.TaskConfig(**t["task_dict"]) for t in slice_tasks]
                seed_here = args.seed + 100000 * epoch + 1000 * b + off

                if slice_tasks:
                    logger.info(
                        "[COLLECT][BASE] chunk off=%d n=%d seed=%d tasks=%s",
                        off, len(slice_tasks), seed_here, _fmt_ids([t["task_id"] for t in slice_tasks]),
                    )

                ro_list = collect_explore_vec(
                    policy_net, cfgs, device, max_steps=250,
                    seed_base=seed_here,
                    dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                )
                for ttask, ro in zip(slice_tasks, ro_list):
                    explore_cache[ttask["task_id"]] = ro
                    if is_task_verbose and debug_shapes:
                        logger.debug(
                            "[COLLECT] tid=%s Tx=%d obs6=%s actions=%s rewards=%s",
                            ttask["task_id"], ro.obs6.shape[0], _shape_str(ro.obs6),
                            _shape_str(ro.actions), _shape_str(ro.rewards),
                        )

            # Build per-task CPU tensors once (threaded), then H2D on copy stream
            per_task_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
            tasks_used: List[int] = []

            # --- INDICATORS holders for this batch ---
            batch_kl_vals, batch_ess_vals = [], []
            batch_rho_mean, batch_rho_std, batch_rho_max, batch_rho_clip_frac = [], [], [], []
            batch_reuse_counts = []
            kl_refresh_count = 0
            ess_refresh_count = 0

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

                # keep on CPU for now; H2D together later
                exp_six_cpu = ro.obs6
                actions_cpu = ro.actions
                rewards_cpu = ro.rewards
                beh_logits_cpu = ro.beh_logits
                Tx = exp_six_cpu.shape[0]

                # KL refresh (+ IS stats) requires logits_now_tmp -> temporary H2D
                with torch.no_grad():
                    logits_now_tmp, _ = policy_net(exp_six_cpu.unsqueeze(0).to(device, non_blocking=True))
                    logits_now_tmp = logits_now_tmp[0] if logits_now_tmp.dim() == 3 else logits_now_tmp
                    lp_cur = torch.log_softmax(logits_now_tmp, dim=-1).gather(1, actions_cpu.to(device, non_blocking=True).unsqueeze(1)).squeeze(1)
                    lp_beh = torch.log_softmax(beh_logits_cpu.to(device, non_blocking=True), dim=-1).gather(1, actions_cpu.to(device, non_blocking=True).unsqueeze(1)).squeeze(1)
                    rhos = torch.exp(lp_cur - lp_beh)
                    ess_ratio_val = ess_ratio_from_rhos(rhos).item()
                    kl_val = mean_kl_logits(logits_now_tmp, beh_logits_cpu.to(device, non_blocking=True)).item()

                batch_kl_vals.append(kl_val)
                batch_ess_vals.append(ess_ratio_val)
                batch_rho_mean.append(float(rhos.mean().item()))
                batch_rho_std.append(float(rhos.std(unbiased=False).item()))
                batch_rho_max.append(float(rhos.max().item()))
                batch_rho_clip_frac.append(_frac_clipped(rhos, args.is_clip_rho) if args.is_clip_rho else 0.0)
                batch_reuse_counts.append(explore_cache[tid].reuse_count if tid in explore_cache else 0)

                if is_task_verbose:
                    logger.info("[KL] tid=%s mean_kl=%.4f thr=%.4f", tid, kl_val, args.kl_refresh_threshold)
                if kl_val > args.kl_refresh_threshold:
                    kl_refresh_count += 1
                    seed_here = args.seed + 999999
                    logger.info("[COLLECT][KL-REFRESH] tid=%s seed=%d (kl=%.4f > %.4f)", tid, seed_here, kl_val, args.kl_refresh_threshold)
                    ro = collect_explore_vec(
                        policy_net, [cfg], device, max_steps=250,
                        seed_base=seed_here,
                        dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                    ).pop()
                    explore_cache[tid] = ro
                    exp_six_cpu = ro.obs6
                    actions_cpu = ro.actions
                    rewards_cpu = ro.rewards
                    beh_logits_cpu = ro.beh_logits
                    Tx = exp_six_cpu.shape[0]
                    logger.info("[KL][REFRESH] tid=%s new_Tx=%d", tid, Tx)
                    with torch.no_grad():
                        logits_now_tmp, _ = policy_net(exp_six_cpu.unsqueeze(0).to(device, non_blocking=True))
                        logits_now_tmp = logits_now_tmp[0] if logits_now_tmp.dim() == 3 else logits_now_tmp
                        lp_cur = torch.log_softmax(logits_now_tmp, dim=-1).gather(1, actions_cpu.to(device, non_blocking=True).unsqueeze(1)).squeeze(1)
                        lp_beh = torch.log_softmax(beh_logits_cpu.to(device, non_blocking=True), dim=-1).gather(1, actions_cpu.to(device, non_blocking=True).unsqueeze(1)).squeeze(1)
                        rhos = torch.exp(lp_cur - lp_beh)
                        ess_ratio_val = ess_ratio_from_rhos(rhos).item()

                # ESS guard (only if reusing explores)
                if args.ess_refresh_ratio and args.ess_refresh_ratio > 0:
                    if is_task_verbose:
                        logger.info("[ESS] tid=%s Tx=%d ess_ratio=%.3f thr=%.3f reuse_count=%d",
                                    tid, Tx, ess_ratio_val, args.ess_refresh_ratio, ro.reuse_count)
                    if int(getattr(args, "explore_reuse_M", 1)) > 1 and ess_ratio_val < args.ess_refresh_ratio:
                        ess_refresh_count += 1
                        seed_here = args.seed + 31337
                        logger.info("[COLLECT][ESS-REFRESH] tid=%s seed=%d (ess=%.3f < %.3f)",
                                    tid, seed_here, ess_ratio_val, args.ess_refresh_ratio)
                        ro = collect_explore_vec(
                            policy_net, [cfg], device, max_steps=250,
                            seed_base=seed_here,
                            dbg=is_task_verbose, dbg_timing=(debug_timing and is_task_verbose), dbg_level=debug_level
                        ).pop()
                        explore_cache[tid] = ro
                        exp_six_cpu = ro.obs6
                        actions_cpu = ro.actions
                        rewards_cpu = ro.rewards
                        beh_logits_cpu = ro.beh_logits
                        Tx = exp_six_cpu.shape[0]
                        logger.info("[ESS][REFRESH] tid=%s new_Tx=%d", tid, Tx)

                # Build batched BC tensors from selected demos (CPU, threaded)
                prev_action_start = float(actions_cpu[-1].item()) if Tx > 0 else 0.0
                selected_paths = _select_demo_paths_for_task(
                    task.get("recs_main", []),
                    task.get("recs_syn", []),
                    args,
                    epoch=epoch,
                )

                if is_task_verbose:
                    logger.info("[SYN] epoch=%d tid=%s using demos: main<=%d syn<=%d selected=%d (syn enabled from epoch %d)",
                                epoch, task["task_id"],
                                int(getattr(args, "max_main_demos_per_task", 0)),
                                int(getattr(args, "max_syn_demos_per_task", 0)),
                                len(selected_paths),
                                int(getattr(args, "syn_demo_min_epoch", 3)))

                def _load_and_build_one(demo_path: str):
                    pre = npz_cache.pop(demo_path, None)
                    p2_six, p2_labels = load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start, preloaded=pre)
                    if p2_six.numel() == 0:
                        return None
                    p2_six = maybe_augment_demo_six_cpu(p2_six, args)
                    obs6_cat, labels_cat = concat_explore_and_exploit(exp_six_cpu, p2_six, p2_labels)
                    return obs6_cat, labels_cat

                demo_obs_list: List[torch.Tensor] = []
                demo_lab_list: List[torch.Tensor] = []
                futures = [cpu_pool.submit(_load_and_build_one, dp) for dp in selected_paths]
                for fut in futures:
                    out = fut.result()
                    if out is None:
                        continue
                    x, y = out
                    demo_obs_list.append(x)
                    demo_lab_list.append(y)

                if len(demo_obs_list) == 0:
                    logger.warning("[BC][SKIP] tid=%s no demos to supervise in this batch", tid)
                    continue

                batch_obs_cpu, batch_lab_cpu = _pad_and_stack_cpu(demo_obs_list, demo_lab_list)
                if getattr(args, "pin_memory", False):
                    batch_obs_cpu = batch_obs_cpu.pin_memory()
                    batch_lab_cpu = batch_lab_cpu.pin_memory()
                    exp_six_cpu = exp_six_cpu.pin_memory()
                    actions_cpu = actions_cpu.pin_memory()
                    rewards_cpu = rewards_cpu.pin_memory()
                    beh_logits_cpu = beh_logits_cpu.pin_memory()

                per_task_tensors[tid] = {
                    "batch_obs_cpu": batch_obs_cpu,
                    "batch_lab_cpu": batch_lab_cpu,
                    "exp_cpu": exp_six_cpu,
                    "actions_cpu": actions_cpu,
                    "rewards_cpu": rewards_cpu,
                    "beh_logits_cpu": beh_logits_cpu,
                    "ro": ro,
                    "task_dict": task["task_dict"],
                }
                tasks_used.append(tid)

            if debug and debug_tasks_per_batch > 0:
                tasks_used = tasks_used[:debug_tasks_per_batch]

            if len(tasks_used) == 0:
                continue

            # ---- Stage H2D copies on a dedicated CUDA stream to overlap with any pending work
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

            # ---- Batch indicator summary (INFO)
            s_kl = _stats_from_list(batch_kl_vals)
            s_ess = _stats_from_list(batch_ess_vals)
            s_rmu = _stats_from_list(batch_rho_mean)
            s_rsd = _stats_from_list(batch_rho_std)
            s_rmx = _stats_from_list(batch_rho_max)
            s_rcl = _stats_from_list(batch_rho_clip_frac)
            s_reu = _stats_from_list(batch_reuse_counts)

            logger.info(
                "[IND][BATCH %d/%d] tasks=%d | %s | thr=%.3g refresh=%d | ESS:%s thr=%.3g refresh=%d | "
                "rho(mean):%s rho(std):%s rho(max):%s rho(clip%%):%s | reuse:%s",
                b + 1, num_batches, len(tasks_used),
                _fmt_stats("KL", s_kl), float(args.kl_refresh_threshold), kl_refresh_count,
                _fmt_stats("", s_ess), float(args.ess_refresh_ratio), ess_refresh_count,
                _fmt_stats("", s_rmu), _fmt_stats("", s_rsd), _fmt_stats("", s_rmx),
                _fmt_stats("", s_rcl), _fmt_stats("", s_reu),
            )

            # -------- Critic auxiliary regression (optional)
            if "critic_aux_steps" in args and args.critic_aux_steps > 0:
                pass  # unchanged

            # =========================
            # OUTER LOOP
            # =========================
            es_eps_bank_cpu: Optional[List[Dict[str, torch.Tensor]]] = None
            for step_idx in range(args.nbc):

                # ----------- ES/SPSA branch (black-box outer) -----------
                if getattr(args, "es_enabled", False):
                    optimizer.zero_grad(set_to_none=True)

                    use_pg_inner = (getattr(args, "es_inner_pg_alpha", 0.0) > 0.0)
                    inner_use_is = (not getattr(args, "es_recollect_inner", False)) and bool(
                        getattr(args, "es_inner_pg_use_is", False)
                    )

                    named_params = select_es_named_params(policy_net, getattr(args, "es_scope", "policy"))

                    # Prepare reusable CPU epsilon bank once per batch if requested
                    if step_idx == 0 and bool(getattr(args, "es_reuse_eps_bank", False)):
                        es_eps_bank_cpu = _sample_eps_bank_cpu(
                            named_params,
                            "spsa" if getattr(args, "es_algo", "es") == "spsa" else "es",
                            int(getattr(args, "es_popsize", 8)),
                            pin=bool(getattr(args, "pin_memory", False)),
                        )

                    # f(θ) baseline
                    with torch.no_grad():
                        bc_phi_base = []
                        for tid in tasks_used:
                            t = per_task_tensors[tid]
                            ro0 = t["ro"]
                            if use_pg_inner:
                                bc_phi0 = meta_objective_with_inner_pg(
                                    policy_net, ro0, t["batch_obs_dev"], t["batch_lab_dev"],
                                    args, device,
                                    alpha=getattr(args, "es_inner_pg_alpha", 0.0),
                                    scope=getattr(args, "es_inner_pg_scope", "head"),
                                    use_is=inner_use_is,
                                    is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                    ess_min_ratio=(
                                        getattr(args, "es_ess_min_ratio", None)
                                        if not getattr(args, "es_recollect_inner", False) else None
                                    ),
                                )
                            else:
                                bc_phi0 = meta_objective_from_rollout(
                                    policy_net, ro0, t["batch_obs_dev"], t["batch_lab_dev"],
                                    args, device,
                                    use_is_inner=(not getattr(args, "es_recollect_inner", False))
                                                  and bool(getattr(args, "es_use_is_inner", False)),
                                    is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                    ess_min_ratio=(
                                        getattr(args, "es_ess_min_ratio", None)
                                        if not getattr(args, "es_recollect_inner", False) else None
                                    ),
                                )
                            bc_phi_base.append(float(bc_phi0.cpu()))
                        if bc_phi_base:
                            running_bc += float(np.mean(bc_phi_base))
                            count_updates += 1
                    f_theta_baseline = float(np.mean(bc_phi_base)) if bc_phi_base else 0.0

                    es_grads = {n: torch.zeros_like(p) for n, p in named_params.items()}
                    vec_cap = int(getattr(args, "num_envs", 8))
                    pop_fplus, pop_fminus, pop_coeffs = [], [], []

                    # If using a CPU eps bank, move once per step to GPU on a copy stream
                    eps_bank_dev: Optional[List[Dict[str, torch.Tensor]]] = None
                    if es_eps_bank_cpu is not None:
                        if device.type == "cuda":
                            eps_bank_dev = []
                            copy_stream = torch.cuda.Stream()
                            with torch.cuda.stream(copy_stream):
                                for eps_cpu in es_eps_bank_cpu:
                                    eps_dev = {n: e.to(device, non_blocking=True) for n, e in eps_cpu.items()}
                                    eps_bank_dev.append(eps_dev)
                            torch.cuda.current_stream().wait_stream(copy_stream)
                        else:
                            eps_bank_dev = es_eps_bank_cpu  # CPU training (unlikely)

                    for i in range(int(getattr(args, "es_popsize", 8))):
                        # Use pre-sampled eps if available; otherwise sample on-the-fly (device-native)
                        eps = None
                        if eps_bank_dev is not None:
                            eps = eps_bank_dev[i]
                        else:
                            eps = sample_eps(named_params, "spsa" if getattr(args, "es_algo", "es") == "spsa" else "es")

                        use_common = bool(getattr(args, "es_common_seed", False))
                        pair_seed = (args.seed + 17_000_000 * epoch + 10_000 * b + 100 * step_idx + i) if use_common else None

                        # f(θ + σ ε)
                        with PerturbContext(named_params, eps, getattr(args, "es_sigma", 0.02), +1):
                            bc_list_plus = []
                            ros_plus_by_tid: Dict[int, ExploreRollout] = {}
                            if getattr(args, "es_recollect_inner", False):
                                seed_for_pair = (pair_seed if pair_seed is not None else args.seed + 99991 + i)
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
                                        policy_net, ro_plus, t["batch_obs_dev"], t["batch_lab_dev"],
                                        args, device,
                                        alpha=getattr(args, "es_inner_pg_alpha", 0.0),
                                        scope=getattr(args, "es_inner_pg_scope", "head"),
                                        use_is=inner_use_is,
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(
                                            getattr(args, "es_ess_min_ratio", None)
                                            if not getattr(args, "es_recollect_inner", False) else None
                                        ),
                                    )
                                else:
                                    bc_phi = meta_objective_from_rollout(
                                        policy_net, ro_plus, t["batch_obs_dev"], t["batch_lab_dev"],
                                        args, device,
                                        use_is_inner=(not getattr(args, "es_recollect_inner", False))
                                                      and bool(getattr(args, "es_use_is_inner", False)),
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(
                                            getattr(args, "es_ess_min_ratio", None)
                                            if not getattr(args, "es_recollect_inner", False) else None
                                        ),
                                    )
                                bc_list_plus.append(bc_phi)
                            bc_list_plus = [x for x in bc_list_plus if torch.isfinite(x)]
                            if bc_list_plus:
                                f_plus = torch.stack(bc_list_plus).mean()
                                if bool(getattr(args, "es_ranknorm", False)):
                                    vals = torch.stack(bc_list_plus)
                                    ranks = torch.argsort(torch.argsort(vals))
                                    f_plus = ((ranks.float() + 0.5) / float(len(vals))).mean()
                            else:
                                f_plus = torch.tensor(f_theta_baseline, device=device)

                        # f(θ - σ ε)
                        with PerturbContext(named_params, eps, getattr(args, "es_sigma", 0.02), -1):
                            bc_list_minus = []
                            ros_minus_by_tid: Dict[int, ExploreRollout] = {}
                            if getattr(args, "es_recollect_inner", False):
                                seed_for_pair = (pair_seed if pair_seed is not None else args.seed + 99991 + i)
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
                                        policy_net, ro_minus, t["batch_obs_dev"], t["batch_lab_dev"],
                                        args, device,
                                        alpha=getattr(args, "es_inner_pg_alpha", 0.0),
                                        scope=getattr(args, "es_inner_pg_scope", "head"),
                                        use_is=inner_use_is,
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(
                                            getattr(args, "es_ess_min_ratio", None)
                                            if not getattr(args, "es_recollect_inner", False) else None
                                        ),
                                    )
                                else:
                                    bc_phi = meta_objective_from_rollout(
                                        policy_net, ro_minus, t["batch_obs_dev"], t["batch_lab_dev"],
                                        args, device,
                                        use_is_inner=(not getattr(args, "es_recollect_inner", False))
                                                      and bool(getattr(args, "es_use_is_inner", False)),
                                        is_clip_rho=getattr(args, "is_clip_rho", 0.0),
                                        ess_min_ratio=(
                                            getattr(args, "es_ess_min_ratio", None)
                                            if not getattr(args, "es_recollect_inner", False) else None
                                        ),
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

                        # ES gradient estimate
                        if bool(getattr(args, "es_fitness_baseline", False)):
                            coeff = ((f_plus - f_theta_baseline) - (f_minus - f_theta_baseline)) / (
                                2.0 * float(getattr(args, "es_sigma", 0.02))
                            )
                        else:
                            coeff = (f_plus - f_minus) / (2.0 * float(getattr(args, "es_sigma", 0.02)))

                        pop_fplus.append(float(f_plus.detach().item()))
                        pop_fminus.append(float(f_minus.detach().item()))
                        pop_coeffs.append(float(coeff.detach().item()))

                        for n, p in named_params.items():
                            es_grads[n] = es_grads[n].add_(coeff * eps[n])

                    # Average and apply
                    for n in es_grads:
                        es_grads[n].div_(float(getattr(args, "es_popsize", 8)))
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            p.grad = None
                    for n, p in named_params.items():
                        p.grad = es_grads[n]
                    if getattr(args, "es_clip_grad", 0.0) and getattr(args, "es_clip_grad", 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(list(named_params.values()), getattr(args, "es_clip_grad", 1.0))
                    with torch.no_grad():
                        es_grad_sq = 0.0
                        for n, p in named_params.items():
                            if p.grad is not None:
                                es_grad_sq += float((p.grad.detach() ** 2).sum().item())
                        es_grad_norm = math.sqrt(es_grad_sq) if es_grad_sq > 0 else 0.0

                    optimizer.step()

                    # housekeeping
                    for tid in tasks_used:
                        explore_cache[tid].reuse_count += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    s_fp = _stats_from_list(pop_fplus)
                    s_fm = _stats_from_list(pop_fminus)
                    s_cf = _stats_from_list(pop_coeffs)
                    logger.info(
                        "[ES][STEP %d/%d] pop=%d σ=%.3f scope=%s inner_pg(alpha=%.3g,scope=%s,use_is=%s,recollect=%s) | "
                        "f+:%s | f-:%s | coeff:%s | grad_norm=%.3e | bc_base=%.4f",
                        step_idx + 1, args.nbc,
                        int(getattr(args, "es_popsize", 8)),
                        float(getattr(args, "es_sigma", 0.02)),
                        str(getattr(args, "es_scope", "policy")),
                        float(getattr(args, "es_inner_pg_alpha", 0.0)),
                        str(getattr(args, "es_inner_pg_scope", "head")),
                        str(inner_use_is),
                        str(bool(getattr(args, "es_recollect_inner", False))),
                        _fmt_stats("", s_fp), _fmt_stats("", s_fm), _fmt_stats("", s_cf),
                        es_grad_norm,
                        f_theta_baseline,
                    )
                    continue  # ES branch finished

                # ----------- Backprop branch (pure-BC outer) -----------
                optimizer.zero_grad(set_to_none=True)
                loss_bc_list = []
                delta_list = []
                inrl_monitor_sum = 0.0

                # plastic diagnostics
                mod_mean, mod_std, mod_min, mod_max, mod_clip_frac = [], [], [], [], []
                delta_vals, delta_pos_count = [], 0

                for idx_task, tid in enumerate(tasks_used):
                    t = per_task_tensors[tid]
                    exp = t["exp_dev"]; acts = t["actions_dev"]; beh = t["beh_logits_dev"]
                    rews_raw = t["rewards_dev"]
                    rews = torch.clamp(rews_raw * args.rew_scale, -args.rew_clip, args.rew_clip)

                    # truncate explore window
                    if args.adapt_trunc_T is not None and isinstance(args.adapt_trunc_T, int) and exp.shape[0] > args.adapt_trunc_T:
                        exp = exp[-args.adapt_trunc_T:]; acts = acts[-args.adapt_trunc_T:]; rews = rews[-args.adapt_trunc_T:]; beh  = beh[-args.adapt_trunc_T:]
                    if is_task_verbose and debug_shapes and idx_task < debug_tasks_per_batch and step_idx == 0:
                        logger.debug("[PLASTIC][TRUNC] tid=%s Tx_used=%d", tid, exp.shape[0])

                    # pass 0: BC at θ
                    with torch.no_grad():
                        if hasattr(policy_net, "reset_plastic"):
                            policy_net.reset_plastic(batch_size=1, device=device)
                            policy_net.set_plastic(update_traces=False, modulators=None)
                        logits_bc_theta, _ = policy_net(t["batch_obs_dev"])
                        if args.label_smoothing > 0.0:
                            loss_bc_theta = smoothed_cross_entropy(
                                logits_bc_theta, t["batch_lab_dev"], ignore_index=PAD_ACTION, smoothing=args.label_smoothing
                            )
                        else:
                            ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_theta = ce(
                                logits_bc_theta.reshape(-1, logits_bc_theta.size(-1)),
                                t["batch_lab_dev"].reshape(-1)
                            )

                    # pass 1: compute modulators m_t
                    if hasattr(policy_net, "reset_plastic"):
                        policy_net.reset_plastic(batch_size=1, device=device)
                        policy_net.set_plastic(update_traces=False, modulators=None)
                    logits_theta, _ = policy_net(exp.unsqueeze(0))
                    logits_theta = logits_theta[0]
                    returns = discounted_returns(rews, args.gamma)
                    T = returns.size(0)
                    ttime = torch.arange(T, device=returns.device, dtype=returns.dtype)
                    zt = (ttime - ttime.mean()) / ttime.std().clamp_min(1e-6)
                    F = torch.stack([torch.ones_like(zt), zt], dim=1)
                    FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
                    w = torch.linalg.solve(FT_F, F.T @ returns)
                    baseline = (F @ w)
                    adv = returns - baseline
                    adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
                    m_t = adv_norm.clamp(-args.plastic_clip_mod, args.plastic_clip_mod)

                    # Optional IS on modulators
                    if getattr(args, "inner_use_is_mod", False):
                        lp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                        lp_beh = torch.log_softmax(beh,           dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                        rho = torch.exp(lp_cur - lp_beh)
                        if args.is_clip_rho and args.is_clip_rho > 0:
                            rho = torch.clamp(rho, max=args.is_clip_rho)
                        m_t = m_t * rho.detach()

                    # collect mod stats
                    mod_mean.append(float(m_t.mean().item()))
                    mod_std.append(float(m_t.std(unbiased=False).item()))
                    mod_min.append(float(m_t.min().item()))
                    mod_max.append(float(m_t.max().item()))
                    mod_clip_frac.append(_frac_at_clip_abs(m_t, getattr(args, "plastic_clip_mod", 0.0)))

                    # pass 2: adapt traces, then BC at φ
                    if hasattr(policy_net, "reset_plastic"):
                        policy_net.reset_plastic(batch_size=1, device=device)
                        policy_net.set_plastic(update_traces=True, modulators=m_t.unsqueeze(0))
                    _ = policy_net(exp.unsqueeze(0))  # update fast weights

                    policy_net.set_plastic(update_traces=False, modulators=None)
                    with autocast(device_type="cuda", enabled=use_amp):
                        logits_phi, _ = policy_net(t["batch_obs_dev"])
                        if args.label_smoothing > 0.0:
                            loss_bc_phi = smoothed_cross_entropy(logits_phi, t["batch_lab_dev"], ignore_index=PAD_ACTION, smoothing=args.label_smoothing)
                        else:
                            ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_phi = ce(logits_phi.reshape(-1, logits_phi.size(-1)), t["batch_lab_dev"].reshape(-1))
                    loss_bc_list.append(loss_bc_phi)

                    delta_k = (loss_bc_theta.detach() - loss_bc_phi.detach())
                    delta_list.append(delta_k)
                    delta_vals.append(float(delta_k.item()))
                    delta_pos_count += int(delta_k.item() > 0.0)

                    # monitor-only PG at θ
                    with torch.no_grad():
                        logp_cur = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                        monitor_pg = - (logp_cur * adv_norm).mean()
                        inrl_monitor_sum += float(monitor_pg.detach().cpu())

                if len(loss_bc_list) == 0:
                    continue

                total_bc = sum(loss_bc_list) / float(len(loss_bc_list))
                if getattr(args, "delta_weight_mode", "none") != "none":
                    deltas = torch.stack(delta_list)
                    dnorm = (deltas - deltas.mean()) / deltas.std().clamp_min(1e-6)
                    if getattr(args, "delta_weight_mode", "none") == "relu":
                        dnorm = torch.relu(dnorm)
                    w = (dnorm - dnorm.min()).detach() + 1e-6
                    w = w / w.sum().clamp_min(1e-6)
                    total_bc = (torch.stack(loss_bc_list) * w).sum()

                avg_loss = total_bc

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
                    torch.cuda.empty_cache()

                s_mod_mean = _stats_from_list(mod_mean)
                s_mod_std = _stats_from_list(mod_std)
                s_mod_min = _stats_from_list(mod_min)
                s_mod_max = _stats_from_list(mod_max)
                s_mod_clip = _stats_from_list(mod_clip_frac)
                s_delta = _stats_from_list(delta_vals)
                frac_pos = (delta_pos_count / max(1, len(delta_vals)))

                logger.info(
                    "[IND][PLASTIC STEP %d/%d] ΔBC:%s (frac>0=%.2f) | m_t: mean:%s std:%s min:%s max:%s clip%%:%s | grad_norm=%.3e",
                    step_idx + 1, args.nbc,
                    _fmt_stats("", s_delta), frac_pos,
                    _fmt_stats("", s_mod_mean), _fmt_stats("", s_mod_std),
                    _fmt_stats("", s_mod_min), _fmt_stats("", s_mod_max),
                    _fmt_stats("", s_mod_clip),
                    float(grad_norm if torch.is_tensor(grad_norm) else grad_norm),
                )

            per_task_tensors.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if count_updates > 0 and not getattr(args, "no_tqdm", False):
                pbar.set_postfix(
                    bc=f"{running_bc / max(1, count_updates):.3f}",
                    inrl=f"{running_inrl / max(1, count_updates):.3f}"
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

    cpu_pool.shutdown(wait=True)
    logger.info("BC meta training complete.")


def _safe_npz_load_obs_acts(path: str) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        d = np.load(path)
        # return raw arrays; conversion to float/long happens later
        return path, d["observations"], d["actions"]
    except Exception as e:
        return path, None, None


if __name__ == "__main__":
    run_training()
