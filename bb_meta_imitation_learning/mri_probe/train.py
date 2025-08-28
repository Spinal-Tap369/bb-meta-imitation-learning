# mri_train/train.py

import os, sys, json, math, random, logging, gc
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
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

def _pad_and_stack_cpu(
    obs_list: List[torch.Tensor],
    lab_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return OrderedDict((n, p) for n, p in module.named_parameters() if p.requires_grad)


def _functional_bc_loss_at_phi(
    model: nn.Module,
    params_phi: Dict[str, torch.Tensor],
    batch_obs: torch.Tensor,
    batch_lab: torch.Tensor,
    smoothing: float,
    use_amp: bool
):
    """Stateless forward at adapted parameters φ to compute BC loss."""
    try:
        with autocast(device_type="cuda", enabled=use_amp):
            logits_phi, _ = _functional_call(model, params_phi, (batch_obs,))
            if smoothing > 0.0:
                loss_bc_phi = smoothed_cross_entropy(
                    logits_phi, batch_lab, ignore_index=PAD_ACTION, smoothing=smoothing
                )
            else:
                ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                loss_bc_phi = ce(
                    logits_phi.reshape(-1, logits_phi.size(-1)),
                    batch_lab.reshape(-1)
                )
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
                    loss_bc_phi = smoothed_cross_entropy(
                        logits_phi, batch_lab, ignore_index=PAD_ACTION, smoothing=smoothing
                    )
                else:
                    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                    loss_bc_phi = ce(
                        logits_phi.reshape(-1, logits_phi.size(-1)),
                        batch_lab.reshape(-1)
                    )
            return loss_bc_phi
        finally:
            model.load_state_dict(sd_backup, strict=False)


# ---------- main ----------

def run_training():
    # Better fragmentation behavior (ideally set before process start)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from .remap import remap_pretrained_state
    args = parse_args()

    # Logging / threads
    chosen_level = (args.log_level or getattr(args, "debug_level", None) or ("DEBUG" if getattr(args, "debug", False) else "INFO")).upper()
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

    # Log inner IS reference choice (defaults to θ_init if missing in config)
    inner_is_ref = str(getattr(args, "inner_is_ref", "theta_init"))
    logger.info("[MRI] inner_use_is=%s | inner_is_ref=%s | inner_ess_min=%.3f",
                str(bool(getattr(args, "inner_use_is", False))),
                inner_is_ref,
                float(getattr(args, "inner_ess_min", 0.0) or 0.0))

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

            # ----------- Base collect (no mid-batch recollects; reuse == nbc) -----------
            reuse_cap = max(1, int(getattr(args, "nbc", 1)))
            need_collect = []
            for task in batch_tasks:
                tid = task["task_id"]
                if tid not in explore_cache or getattr(explore_cache[tid], "reuse_count", 0) >= reuse_cap:
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
            # PASS 1 — single stacked forward for metrics and FREE θ_init caching
            kls, esss, rmu, rsd, rmx = [], [], [], [], []

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
                logits_now_b, _ = policy_net(exp_b)  # (B, T_max, A) — THIS IS θ_init

            # Metrics + FREE cache of lp_init (per tid, CPU)
            lp_init_by_tid: Dict[int, torch.Tensor] = {}
            for i, tid in enumerate(tids_order):
                tlen = lens[i]
                if tlen == 0:
                    kls.append(0.0); esss.append(1.0); rmu.append(1.0); rsd.append(0.0); rmx.append(1.0)
                    lp_init_by_tid[tid] = torch.empty((0,), device="cpu")
                    continue
                li = logits_now_b[i, :tlen]
                bi = beh_b[i, :tlen]
                ai = act_b[i, :tlen]

                lp_now = torch.log_softmax(li, dim=-1).gather(1, ai.unsqueeze(1)).squeeze(1)  # θ_init
                lp_beh = torch.log_softmax(bi, dim=-1).gather(1, ai.unsqueeze(1)).squeeze(1)  # behavior

                rhos = torch.exp(lp_now - lp_beh)
                if getattr(args, "is_clip_rho", 0.0) > 0:
                    rhos = torch.clamp(rhos, max=args.is_clip_rho)

                ess_ratio_val = ess_ratio_from_rhos(rhos).item()
                kl_val = mean_kl_logits(li, bi).item()
                kls.append(kl_val); esss.append(ess_ratio_val)
                rmu.append(float(rhos.mean().item()))
                rsd.append(float(rhos.std(unbiased=False).item()))
                rmx.append(float(rhos.max().item()))

                # FREE θ_init ref for this tid (store on CPU; move later)
                lp_init_by_tid[tid] = lp_now.detach().cpu()

            # free PASS-1 tensors
            del exp_b, beh_b, act_b, logits_now_b
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # PASS 2 — build per-task BC supervision tensors (no in-batch recollects)
            per_task_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
            tasks_used: List[int] = []

            for task in batch_tasks:
                tid = task["task_id"]
                ro = explore_cache.get(tid)

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
                    batch_obs_cpu = batch_obs_cpu.pin_memory()
                    batch_lab_cpu = batch_lab_cpu.pin_memory()

                per_task_tensors[tid] = {
                    "batch_obs_cpu": batch_obs_cpu,
                    "batch_lab_cpu": batch_lab_cpu,
                    "exp_cpu": exp_six_cpu,
                    "actions_cpu": actions_cpu,
                    "rewards_cpu": rewards_cpu,
                    "beh_logits_cpu": beh_logits_cpu,
                    "task_dict": task["task_dict"],
                    "selected_paths": selected_paths,
                }
                tasks_used.append(tid)

            if len(tasks_used) == 0:
                continue

            # H2D copies (and move FREE θ_init cache to device)
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
                        # FREE θ_init ref
                        cached = lp_init_by_tid.get(tid, torch.empty((0,)))
                        t["lp_init_dev"] = cached.to(device, non_blocking=True) if cached.numel() > 0 else torch.empty((0,), device=device)
                torch.cuda.current_stream().wait_stream(copy_stream)

            # Batch indicators (from PASS 1)
            s_kl  = _stats_from_list(kls)
            s_ess = _stats_from_list(esss)
            s_rmu = _stats_from_list(rmu)
            s_rsd = _stats_from_list(rsd)
            s_rmx = _stats_from_list(rmx)
            logger.info(
                "[IND][BATCH %d/%d] tasks=%d | %s | ESS:%s | rho(mean):%s rho(std):%s rho(max):%s",
                b + 1, num_batches, len(tasks_used),
                _fmt_stats("KL", s_kl), _fmt_stats("", s_ess),
                _fmt_stats("", s_rmu), _fmt_stats("", s_rsd), _fmt_stats("", s_rmx),
            )

            # =========================
            # MRI OUTER LOOP (no mid-batch recollects)
            # =========================
            # (micro-opt) names outside step loop
            named_all = _named_trainable(policy_net)
            adapt_names = [n for n in named_all.keys() if n not in _critic_param_names(policy_net)]

            for step_idx in range(args.nbc):
                optimizer.zero_grad(set_to_none=True)

                tasks_n = len(tasks_used)
                total_bc_val = 0.0
                total_corr_val = 0.0
                pg_vals = []
                ess_skips = 0  # count of tasks skipped by ESS gate

                # running baseline if 'batch' is requested
                batch_mean_bc = 0.0
                batch_seen = 0

                for tid in tasks_used:
                    t = per_task_tensors[tid]
                    exp = t["exp_dev"]; acts = t["actions_dev"]; beh = t["beh_logits_dev"]
                    rews_raw = t["rewards_dev"]
                    rews = torch.clamp(rews_raw * args.rew_scale, -args.rew_clip, args.rew_clip)

                    # truncate explore window if needed (keep θ_init cache aligned)
                    adapt_T = getattr(args, "adapt_trunc_T", None)
                    if isinstance(adapt_T, int) and adapt_T > 0 and exp.shape[0] > adapt_T:
                        Tkeep = adapt_T
                        exp = exp[-Tkeep:]; acts = acts[-Tkeep:]; rews = rews[-Tkeep:]; beh  = beh[-Tkeep:]
                        if "lp_init_dev" in t and t["lp_init_dev"].shape[0] > 0:
                            t["lp_init_dev"] = t["lp_init_dev"][-Tkeep:]

                    # ----- Inner policy gradient (with ESS gate) -----
                    params_k = OrderedDict((n, p) for n, p in named_all.items())
                    alpha = float(getattr(args, "inner_pg_alpha", 0.1))
                    ess_min = float(getattr(args, "inner_ess_min", 0.0) or 0.0)

                    # one (or few) inner steps
                    do_skip = False
                    for k in range(int(getattr(args, "inner_steps", 1))):
                        with autocast(device_type="cuda", enabled=use_amp):
                            logits_seq, _ = _functional_call(policy_net, params_k, (exp.unsqueeze(0),))
                            logits_seq = logits_seq[0]

                        # advantages with a tiny linear baseline
                        returns = discounted_returns(rews, args.gamma)
                        Tlen = returns.size(0)
                        tt = torch.arange(Tlen, device=returns.device, dtype=returns.dtype)
                        zt = (tt - tt.mean()) / tt.std().clamp_min(1e-6)
                        F = torch.stack([torch.ones_like(zt), zt], dim=1)
                        FT_F = F.T @ F + 1e-3 * torch.eye(2, device=returns.device, dtype=returns.dtype)
                        w = torch.linalg.solve(FT_F, F.T @ returns)
                        baseline = (F @ w)
                        adv = returns - baseline
                        adv_norm = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

                        logp_cur = torch.log_softmax(logits_seq, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)

                        if bool(getattr(args, "inner_use_is", False)):
                            # θ_init reference (or fallback to behavior if empty)
                            if t["lp_init_dev"].numel() > 0:
                                lp_ref = t["lp_init_dev"]
                            else:
                                lp_ref = torch.log_softmax(beh, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
                            rho = torch.exp(logp_cur.detach() - lp_ref)
                            if getattr(args, "is_clip_rho", 0.0) > 0:
                                rho = torch.clamp(rho, max=args.is_clip_rho)

                            # ---- ESS gate (skip inner update if too low)
                            if ess_min > 0.0:
                                ess_now = ess_ratio_from_rhos(rho)
                                if ess_now.item() < ess_min:
                                    ess_skips += 1
                                    pg_vals.append(0.0)
                                    # Skip all inner updates for this task (φ=θ)
                                    del logits_seq, returns, adv, adv_norm, logp_cur
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                    do_skip = True
                                    break

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

                    # ----- Meta-descent correction term (score-function, lower-variance) -----
                    corr_loss = 0.0
                    if bool(getattr(args, "meta_corr", False)):
                        with autocast(device_type="cuda", enabled=use_amp):
                            # IMPORTANT: use θ (outer params), not φ. policy_net is at θ here.
                            logits_theta, _ = policy_net(exp.unsqueeze(0))
                            logits_theta = logits_theta[0]
                        logp_theta = torch.log_softmax(logits_theta, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)

                        # --- Importance weights for correction (θ / reference), DETACHED ---
                        w_imp = None
                        if bool(getattr(args, "meta_corr_use_is", False)):
                            # Prefer θ_init reference if available; else fall back to behavior.
                            if str(getattr(args, "inner_is_ref", "theta_init")) == "theta_init" and t.get("lp_init_dev", None) is not None and t["lp_init_dev"].numel() > 0:
                                lp_ref = t["lp_init_dev"]
                            else:
                                lp_ref = torch.log_softmax(beh, dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)

                            w_imp = torch.exp(logp_theta.detach() - lp_ref)        # detach reference path
                            if getattr(args, "is_clip_rho", 0.0) > 0:
                                w_imp = torch.clamp(w_imp, max=args.is_clip_rho)

                            # Optional ESS gate for meta-corr (reuse inner_ess_min if no dedicated flag)
                            ess_min_corr = float(getattr(args, "meta_corr_ess_min", getattr(args, "inner_ess_min", 0.0)) or 0.0)
                            if ess_min_corr > 0.0:
                                if ess_ratio_from_rhos(w_imp).item() < ess_min_corr:
                                    # Skip correction when weights are too degenerate.
                                    w_imp = None

                        # --- Center log-probs (variance reduction) ---
                        if bool(getattr(args, "meta_corr_center_logp", False)):
                            if w_imp is not None:
                                wmean = ((w_imp * logp_theta).sum() / (w_imp.sum() + 1e-8)).detach()
                            else:
                                wmean = (logp_theta.mean()).detach()
                            logp_theta = logp_theta - wmean

                        # --- Weighted score; normalize by weight mass for stable scale ---
                        if w_imp is not None:
                            score = (w_imp.detach() * logp_theta).sum() / (w_imp.sum().detach() + 1e-8)
                        else:
                            score = logp_theta.mean()

                        coeff = float(getattr(args, "meta_corr_coeff", 1.0))

                        # Baseline for (L_BC - b) — you already maintain batch/EMA outside
                        if str(getattr(args, "meta_corr_baseline", "batch")) == "ema":
                            bval = float(ema_baseline) if (ema_baseline is not None) else 0.0
                        elif str(getattr(args, "meta_corr_baseline", "batch")) == "batch":
                            bval = float(batch_mean_bc) if batch_seen > 0 else 0.0
                        else:
                            bval = 0.0

                        # NOTE: detach L_BC so gradients only flow through logπ_θ
                        corr_loss = - coeff * (loss_bc_phi.detach() - bval) * score

                        del logits_theta, logp_theta


                    # ---- Backward (streamed)
                    task_loss = loss_bc_phi + (corr_loss if bool(getattr(args, "meta_corr", False)) else 0.0)
                    scaler.scale(task_loss / tasks_n).backward()

                    # update baselines
                    if str(getattr(args, "meta_corr_baseline", "batch")) == "ema":
                        if ema_baseline is None:
                            ema_baseline = bc_scalar_detached
                        else:
                            ema_beta = float(getattr(args, "meta_corr_ema_beta", 0.9))
                            ema_baseline = ema_beta * ema_baseline + (1.0 - ema_beta) * bc_scalar_detached
                    else:
                        batch_seen += 1
                        batch_mean_bc = ((batch_seen - 1) * batch_mean_bc + bc_scalar_detached) / max(1, batch_seen)

                    # bookkeeping: count reuse
                    try:
                        rc = getattr(explore_cache[tid], "reuse_count", 0)
                        explore_cache[tid].reuse_count = rc + 1
                    except Exception:
                        pass

                    del params_k, loss_bc_phi

                # one optimizer step for the whole step_idx
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer); scaler.update()
                else:
                    logger.warning("[MRI][NONFINITE] grad_norm=%s — skipping optimizer step", str(grad_norm))
                    optimizer.zero_grad(set_to_none=True); scaler.update()

                running_bc += total_bc_val / max(1, tasks_n)
                running_pg += (np.mean(pg_vals) if pg_vals else 0.0)
                running_corr += (total_corr_val / max(1, tasks_n) if bool(getattr(args, "meta_corr", False)) else 0.0)
                count_updates += 1

                logger.info(
                    "[MRI][STEP %d/%d] outer_BC=%.4f | inner_pg(mean)=%.4f | corr_loss=%.4f | grad_norm=%.3e | ess_skips=%d",
                    step_idx + 1, args.nbc,
                    float(total_bc_val / max(1, tasks_n)),
                    float(np.mean(pg_vals) if pg_vals else 0.0),
                    float(total_corr_val / max(1, tasks_n) if bool(getattr(args, "meta_corr", False)) else 0.0),
                    float(grad_norm if torch.is_tensor(grad_norm) else grad_norm),
                    int(ess_skips),
                )

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
