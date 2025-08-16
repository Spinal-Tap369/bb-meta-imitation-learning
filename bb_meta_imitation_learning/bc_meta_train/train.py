# bc_meta_train/train.py

import os
import sys
import json
import math
import random
import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import gymnasium as gym

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
    build_six_from_demo_sequence,
    apply_brightness_contrast,
    apply_gaussian_noise,
    apply_spatial_jitter,
)
from .rl_loss import reinforce_with_baseline, mean_kl_logits, ess_ratio_from_rhos
from .phase1_shaping import Phase1ShapingWrapper

logger = logging.getLogger(__name__)

# ---------- logging setup ----------
def _setup_logging():
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

# ---------- explore rollout cache ----------
@dataclass
class ExploreRollout:
    # IMPORTANT: keep cached tensors on CPU to avoid VRAM growth
    obs6: torch.Tensor           # (T,6,H,W)  (CPU)
    actions: torch.Tensor        # (T,)       (CPU)
    rewards: torch.Tensor        # (T,)       (CPU)
    beh_logits: torch.Tensor     # (T,A)      (CPU)
    reuse_count: int

def _collect_explore(
    policy_net,
    env,
    task_cfg,
    device,
    max_steps: int = 250,
) -> ExploreRollout:
    """Run phase-1 on-policy in the env and return tensors for RL training."""
    env.unwrapped.set_task(task_cfg)

    obs, _ = env.reset()
    done = False
    trunc = False

    # infer spatial size from first obs
    H, W = obs.shape[0], obs.shape[1]

    # preallocate compute buffers on GPU (fast), but we'll .cpu() before caching
    states_buf = torch.empty((max_steps, 6, H, W), device=device, dtype=torch.float32)
    beh_logits_buf: List[torch.Tensor] = []
    actions_list: List[int] = []
    rewards_list: List[float] = []

    last_a, last_r = 0.0, 0.0
    steps = 0
    reached = False

    with torch.inference_mode():
        while not done and not trunc and steps < max_steps:
            img = torch.from_numpy(obs.transpose(2, 0, 1)).to(device=device, dtype=torch.float32) / 255.0
            pa = torch.full((1, H, W), last_a, device=device, dtype=torch.float32)
            pr = torch.full((1, H, W), last_r, device=device, dtype=torch.float32)
            bb = torch.zeros((1, H, W), device=device, dtype=torch.float32)  # explore -> 0
            obs6_t = torch.cat([img, pa, pr, bb], dim=0)
            states_buf[steps] = obs6_t

            seq = states_buf[: steps + 1].unsqueeze(0)  # (1, t, 6,H,W)
            logits, _ = policy_net.act_single_step(seq)
            logits_t = logits[0] if logits.dim() == 2 else logits
            beh_logits_buf.append(logits_t.detach().clone())

            action = torch.distributions.Categorical(logits=logits_t).sample().item()
            obs, rew, done, trunc, info = env.step(action)

            # goal reached check (if env exposes phase_metrics)
            if getattr(env.unwrapped, "maze_core", None) is not None:
                try:
                    if env.unwrapped.maze_core.phase_metrics[2]["goal_rewards"] > 0:
                        reached = True
                except Exception:
                    pass

            actions_list.append(action)
            rewards_list.append(float(rew))
            last_a, last_r = float(action), float(rew)
            steps += 1
            if reached:
                break

    # Slice to actual length; MOVE TO CPU for caching
    obs6_dev = states_buf[:steps].contiguous()
    actions_dev = torch.tensor(actions_list, device=device, dtype=torch.long)
    rewards_dev = torch.tensor(rewards_list, device=device, dtype=torch.float32)
    beh_logits_dev = torch.stack(beh_logits_buf, dim=0)

    obs6 = obs6_dev.cpu()
    actions = actions_dev.cpu()
    rewards = rewards_dev.cpu()
    beh_logits = beh_logits_dev.cpu()

    # Explicitly free GPU side (compute buffers)
    del states_buf, obs6_dev, actions_dev, rewards_dev, beh_logits_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ExploreRollout(obs6=obs6, actions=actions, rewards=rewards, beh_logits=beh_logits, reuse_count=0)

def _first_demo_paths(ex_record_list: List[Dict], demo_root: str) -> List[str]:
    demos_p2 = [r for r in ex_record_list if int(r.get("phase", 2)) == 2]
    demos_p2 = sorted(demos_p2, key=lambda r: r["frames"])
    return [os.path.join(demo_root, r["file"]) for r in demos_p2]

# ---------- manifest/trial start-goal consistency ----------
def _manifest_p2_pairs(recs: List[Dict]) -> List[Tuple[int, int, int, int]]:
    pairs = []
    for r in recs:
        if int(r.get("phase", 2)) != 2:
            continue
        sx, sy = r.get("start_x"), r.get("start_y")
        gx, gy = r.get("goal_x"), r.get("goal_y")
        if sx is None or sy is None or gx is None or gy is None:
            continue
        pairs.append((int(sx), int(sy), int(gx), int(gy)))
    return pairs

def _assert_start_goal_match(recs: List[Dict], tdict: Dict, tid: int):
    pairs = _manifest_p2_pairs(recs)
    if not pairs:
        return
    uniq = {pairs[0]}
    for p in pairs[1:]:
        uniq.add(p)
    if len(uniq) > 1:
        raise RuntimeError(
            f"[TASK {tid}] Demo manifest contains multiple (start,goal) pairs for phase-2: {sorted(list(uniq))}"
        )
    msx, msy, mgx, mgy = pairs[0]
    ts = tuple(tdict.get("start", (None, None)))
    tg = tuple(tdict.get("goal", (None, None)))
    if ts != (msx, msy) or tg != (mgx, mgy):
        raise RuntimeError(
            f"[TASK {tid}] train_trials.json start/goal {ts}->{tg} do not match manifest {(msx, msy)}->{(mgx, mgy)}"
        )

def _load_phase2_six_and_labels(demo_path: str, prev_action_start: float) -> Tuple[torch.Tensor, torch.Tensor]:
    d = np.load(demo_path)
    obs = d["observations"].astype(np.float32) / 255.0
    acts = d["actions"].astype(np.int64)
    L = len(acts)
    boundary = np.zeros((L,), dtype=np.float32)
    six = build_six_from_demo_sequence(obs, acts, boundary, prev_action_start=prev_action_start)
    return torch.from_numpy(six).float(), torch.from_numpy(acts).long()

def _concat_explore_and_exploit(
    explore_six: torch.Tensor,       # (Tx,6,H,W)
    exploit_six: torch.Tensor,       # (Te,6,H,W)
    exploit_labels: torch.Tensor,    # (Te,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Tx = explore_six.shape[0]
    Te = exploit_six.shape[0]

    exploit_six = exploit_six.clone()
    if Te > 0:
        exploit_six[:, 5, :, :] = 1.0

    obs6_cat = torch.cat([explore_six, exploit_six], dim=0)        # (T,6,H,W)
    labels_cat = torch.cat(
        [torch.full((Tx,), PAD_ACTION, dtype=torch.long, device=explore_six.device), exploit_labels],
        dim=0
    )
    exp_mask = torch.cat([torch.ones(Tx, device=explore_six.device), torch.zeros(Te, device=explore_six.device)], dim=0)
    val_mask = torch.ones(Tx + Te, device=explore_six.device)
    return obs6_cat, labels_cat, exp_mask, val_mask

def _make_base_env():
    """Create a raw MetaMazeDiscrete3D env (no shaping)."""
    try:
        env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
    except ModuleNotFoundError:
        import importlib
        try:
            pkg = importlib.import_module("bb_meta_imitation_learning.env")
            sys.modules.setdefault("env", pkg)
            sys.modules.setdefault("env.maze_env", importlib.import_module("bb_meta_imitation_learning.env.maze_env"))
            env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
        except Exception:
            from bb_meta_imitation_learning.env.maze_env import MetaMazeDiscrete3D
            env = MetaMazeDiscrete3D(enable_render=False)
    return env

def _maybe_augment_demo_six_cpu(p2_six_cpu: torch.Tensor, args) -> torch.Tensor:
    """
    p2_six_cpu: (T,6,H,W) on CPU, float in [0,1] for RGB channels.
    Only augment channels [0:3] (RGB). Action/reward/boundary left untouched.
    """
    if not bool(getattr(args, "use_aug", True)):
        return p2_six_cpu
    if np.random.rand() > float(getattr(args, "aug_prob", 0.5)):
        return p2_six_cpu

    b_rng = getattr(args, "aug_brightness_range", (0.9, 1.1))
    c_rng = getattr(args, "aug_contrast_range",   (0.9, 1.1))
    noise_std   = float(getattr(args, "aug_noise_std", 0.02))
    jitter_max  = int(getattr(args, "aug_jitter_max", 2))
    p_bc        = float(getattr(args, "aug_bc_prob",   0.5))
    p_noise     = float(getattr(args, "aug_noise_prob", 0.25))
    p_jitter    = float(getattr(args, "aug_jitter_prob", 0.25))

    x = p2_six_cpu.numpy()   # (T,6,H,W)
    imgs = x[:, 0:3, :, :]

    if np.random.rand() < p_bc:
        imgs = apply_brightness_contrast(imgs, brightness_range=b_rng, contrast_range=c_rng)
    if np.random.rand() < p_noise:
        imgs = apply_gaussian_noise(imgs, std=noise_std)
    if np.random.rand() < p_jitter:
        imgs = apply_spatial_jitter(imgs, max_shift=jitter_max)

    x[:, 0:3, :, :] = imgs
    return torch.from_numpy(x).float()

def run_training():
    args = parse_args()
    _setup_logging()
    start_time = datetime.datetime.utcnow()

    # —— New knobs (safe defaults if not present in config) ——
    nbc = int(getattr(args, "nbc", 8))  # number of outer updates per collected explore
    bc_only_after_first = bool(getattr(args, "bc_only_after_first", False))
    recompute_rl_each_outer = bool(getattr(args, "recompute_rl_each_outer", True))

    # Seeding
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.load_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Load tasks
    with open(args.train_trials) as f:
        tasks_all = json.load(f)
    task_index_to_dict = {i: t for i, t in enumerate(tasks_all)}
    task_hash_to_dict = {make_task_id(t): t for t in tasks_all}

    # Load demo manifests
    demos_by_task = load_all_manifests(args.demo_root)
    if not demos_by_task:
        raise RuntimeError("No demo manifests found in demo_root.")

    # Build task list with phase-2 demo paths (use all)
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

        _assert_start_goal_match(recs, tdict, tid)
        p2_paths = _first_demo_paths(recs, args.demo_root)
        if len(p2_paths) == 0:
            continue
        tasks.append({"task_id": tid, "task_dict": tdict, "p2_paths": p2_paths})

    if not tasks:
        raise RuntimeError("No tasks with phase-2 demos found.")
    random.shuffle(tasks)
    n_val = min(len(tasks), args.val_size)
    val_tasks = tasks[:n_val]
    train_tasks = tasks[n_val:]
    logger.info(f"[SPLIT] train={len(train_tasks)} val={len(val_tasks)}")

    # Model
    policy_net = build_model(seq_len=SEQ_LEN).to(device)

    # BC init
    if args.bc_init:
        ck_path = os.path.abspath(args.bc_init)
        if not os.path.isfile(ck_path):
            raise FileNotFoundError(f"{ck_path} not found")
        sd = torch.load(ck_path, map_location="cpu")
        ret = policy_net.load_state_dict(sd, strict=False)
        miss = getattr(ret, "missing_keys", []); unex = getattr(ret, "unexpected_keys", [])
        logger.info(f"[INIT] loaded BC init from {ck_path} (missing={len(miss)}, unexpected={len(unex)})")

    # Optimizer with encoder schedule
    def _find_encoder_module(net: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
        cand = ["image_encoder", "cnn", "conv_frontend", "visual_encoder", "encoder", "backbone"]
        for name in cand:
            mod = getattr(net, name, None)
            if isinstance(mod, torch.nn.Module):
                return mod, name
        for name, mod in net.named_children():
            lname = name.lower()
            if any(k in lname for k in ["enc", "cnn", "conv", "vision", "image"]) and isinstance(mod, torch.nn.Module):
                return mod, name
        return None, None

    encoder_module, enc_name = _find_encoder_module(policy_net)
    if encoder_module is None:
        logger.warning("No visual encoder detected; single param group.")
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        have_encoder = False
    else:
        have_encoder = True
        logger.info(f"[ENCODER] Using submodule '{enc_name}'.")
        warmup = max(0, int(args.freeze_encoder_warmup_epochs))
        enc_params = list(encoder_module.parameters())
        enc_ids = {id(p) for p in enc_params}
        rest_params = [p for p in policy_net.parameters() if id(p) not in enc_ids]
        if warmup > 0:
            for p in encoder_module.parameters(): p.requires_grad = False
            optimizer = torch.optim.Adam(rest_params, lr=args.lr, weight_decay=args.weight_decay)
            logger.info(f"[ENCODER] Frozen for first {warmup} epoch(s).")
        else:
            for p in encoder_module.parameters(): p.requires_grad = True
            optimizer = torch.optim.Adam(
                [
                    {"params": rest_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": enc_params,  "lr": args.lr * args.encoder_lr_mult, "weight_decay": args.weight_decay},
                ]
            )
            logger.info(f"[ENCODER] No warmup; encoder lr mult = {args.encoder_lr_mult}.")

    # Resume (model-only)
    start_epoch, best_val_score = load_checkpoint(policy_net, args.load_path)
    best_epoch = start_epoch
    logger.info(f"[RESUME] from epoch {start_epoch + 1}")

    # Envs: wrapped train env (shaping on), plain val env (no shaping)
    train_env = _make_base_env()
    train_env = Phase1ShapingWrapper(train_env)  # shaping only influences phase-1
    train_env.action_space.seed(args.seed)

    val_env = _make_base_env()
    val_env.action_space.seed(args.seed)

    patience = 0
    improved = False
    final_epoch = start_epoch

    free_every = 8  # call empty_cache every N updates (CUDA only)
    updates_since_free = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        final_epoch = epoch

        # Encoder unfreeze boundary
        if have_encoder and args.freeze_encoder_warmup_epochs > 0 and epoch == (args.freeze_encoder_warmup_epochs + 1):
            for p in encoder_module.parameters(): p.requires_grad = True
            enc_params = list(encoder_module.parameters())
            enc_ids = {id(p) for p in enc_params}
            rest_params = [p for p in policy_net.parameters() if id(p) not in enc_ids]
            optimizer = torch.optim.Adam(
                [
                    {"params": rest_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": enc_params,  "lr": args.lr * args.encoder_lr_mult, "weight_decay": args.weight_decay},
                ]
            )
            logger.info("[ENCODER] Warmup complete; switched to two param groups.")

        # ---------- epoch-local caches ----------
        explore_cache: Dict[int, ExploreRollout] = {}  # CPU rollouts
        # Per-batch tensors keyed by tid; cleared every minibatch
        per_task_tensors: Dict[int, Dict[str, torch.Tensor]] = {}

        num_batches = math.ceil(len(train_tasks) / max(1, args.batch_size))
        batch_indices = list(range(len(train_tasks)))
        random.shuffle(batch_indices)

        policy_net.train()
        running_train_loss = 0.0
        running_bc = 0.0
        running_rl = 0.0
        running_ent = 0.0
        count_updates = 0

        pbar = tqdm(range(num_batches), desc=f"[Epoch {epoch:02d}] train", leave=False)
        for b in pbar:
            start = b * args.batch_size
            end = min(len(train_tasks), (b + 1) * args.batch_size)
            batch_ids = [batch_indices[i] for i in range(start, end)]
            batch_tasks = [train_tasks[i] for i in batch_ids]

            # --------- PREPARE each task in the minibatch (build tensors once) ---------
            per_task_tensors.clear()
            tasks_used: List[int] = []

            for task in batch_tasks:
                tid = task["task_id"]
                cfg = MazeTaskManager.TaskConfig(**task["task_dict"])

                # Ensure we have a fresh or valid explore rollout (cache lives on CPU)
                need_new = (tid not in explore_cache) or (explore_cache[tid].reuse_count >= args.explore_reuse_M)
                if need_new:
                    rollout = _collect_explore(policy_net, train_env, cfg, device, max_steps=250)
                    explore_cache[tid] = rollout
                else:
                    # KL-based refresh against current policy (copy CPU->GPU just for check)
                    with torch.no_grad():
                        seq_dev_tmp = explore_cache[tid].obs6.to(device, non_blocking=True).unsqueeze(0)
                        logits_all_tmp, _ = policy_net(seq_dev_tmp)  # (1,Tx,A)
                        cur_logits_tmp = logits_all_tmp[0] if logits_all_tmp.dim() == 3 else logits_all_tmp
                        if mean_kl_logits(cur_logits_tmp, explore_cache[tid].beh_logits.to(device)).item() > args.kl_refresh_threshold:
                            rollout = _collect_explore(policy_net, train_env, cfg, device, max_steps=250)
                            explore_cache[tid] = rollout
                    try:
                        del seq_dev_tmp, logits_all_tmp, cur_logits_tmp
                    except NameError:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # Copy explore chunk to device
                exp_six_dev  = explore_cache[tid].obs6.to(device, non_blocking=True)       # (Tx,6,H,W)
                actions_x    = explore_cache[tid].actions.to(device, non_blocking=True)    # (Tx,)
                rewards_x    = explore_cache[tid].rewards.to(device, non_blocking=True)    # (Tx,)
                beh_logits_x = explore_cache[tid].beh_logits.to(device, non_blocking=True) # (Tx,A)
                Tx = exp_six_dev.shape[0]

                # RL precompute at k==0 baseline
                with autocast(device_type="cuda", enabled=use_amp):
                    logits_x_all, values_x_all = policy_net(exp_six_dev.unsqueeze(0))
                    cur_logits_x0 = logits_x_all[0] if logits_x_all.dim() == 3 else logits_x_all
                    values_x0 = values_x_all
                    if values_x0.dim() == 3:
                        values_x0 = values_x0[0, :, 0]
                    elif values_x0.dim() == 2:
                        values_x0 = values_x0[0]
                    else:
                        values_x0 = values_x0.squeeze()

                    loss_rl_0, ent_x_0, _ = reinforce_with_baseline(
                        cur_logits=cur_logits_x0,
                        actions=actions_x,
                        rewards=rewards_x,
                        values=values_x0,
                        gamma=args.gamma,
                        lam=args.gae_lambda,
                        use_gae=args.use_gae,
                        entropy_coef=args.explore_entropy_coef,
                        behavior_logits=beh_logits_x if args.offpolicy_correction != "none" else None,
                        offpolicy=args.offpolicy_correction,
                        is_clip_rho=args.is_clip_rho,
                    )

                # Build BC tensors for ALL demos (once per task)
                prev_action_start = float(actions_x[-1].item()) if Tx > 0 else 0.0
                demo_obs_list: List[torch.Tensor] = []
                demo_lab_list: List[torch.Tensor] = []

                for demo_path in task["p2_paths"]:
                    p2_six, p2_labels = _load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start)
                    if p2_six.numel() == 0:
                        continue
                    p2_six = _maybe_augment_demo_six_cpu(p2_six, args)
                    p2_six = p2_six.to(device, non_blocking=True)
                    p2_labels = p2_labels.to(device, non_blocking=True)
                    obs6_cat, labels_cat, _, _ = _concat_explore_and_exploit(exp_six_dev, p2_six, p2_labels)
                    demo_obs_list.append(obs6_cat)
                    demo_lab_list.append(labels_cat)

                if len(demo_obs_list) == 0:
                    # Nothing to supervise for this task — skip it this batch
                    # Free heavy tensors from this task before continuing
                    del exp_six_dev, actions_x, rewards_x, beh_logits_x
                    try:
                        del logits_x_all, values_x_all, cur_logits_x0, values_x0
                    except NameError:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                T_max = max(x.shape[0] for x in demo_obs_list)
                H, W = demo_obs_list[0].shape[-2], demo_obs_list[0].shape[-1]
                pad_obs, pad_lab = [], []
                for x, y in zip(demo_obs_list, demo_lab_list):
                    t = x.shape[0]
                    if t < T_max:
                        pad_t = T_max - t
                        x = torch.cat([torch.zeros((pad_t, 6, H, W), device=device, dtype=x.dtype), x], dim=0)
                        y = torch.cat([torch.full((pad_t,), PAD_ACTION, device=device, dtype=y.dtype), y], dim=0)
                    pad_obs.append(x); pad_lab.append(y)

                batch_obs = torch.stack(pad_obs, dim=0)  # (B_d, T_max, 6,H,W)
                batch_lab = torch.stack(pad_lab, dim=0)  # (B_d, T_max)

                # Stash everything we need per task for the k-loop
                per_task_tensors[tid] = {
                    "batch_obs": batch_obs,
                    "batch_lab": batch_lab,
                    "loss_rl0": loss_rl_0,             # graph-bearing if recompute=False and k==0
                    "ent0": torch.as_tensor(ent_x_0).detach(),
                }
                if recompute_rl_each_outer:
                    per_task_tensors[tid].update({
                        "exp": exp_six_dev,
                        "actions": actions_x,
                        "rewards": rewards_x,
                        "beh_logits": beh_logits_x,
                    })
                else:
                    # Free big explore tensors if we won't recompute RL each k
                    del exp_six_dev, actions_x, rewards_x, beh_logits_x
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                tasks_used.append(tid)

            # --------- OUTER LOOP: MRI-style (one step per k over the batch) ---------
            if len(tasks_used) == 0:
                # No supervised data in this minibatch; skip optimizer work
                continue

            for k in range(nbc):
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                total_bc = 0.0
                total_rl = 0.0
                total_ent = 0.0
                denom = 0

                for tid in tasks_used:
                    tensors = per_task_tensors[tid]
                    batch_obs = tensors["batch_obs"]
                    batch_lab = tensors["batch_lab"]

                    # --- RL term selection ---
                    if bc_only_after_first:
                        if k == 0:
                            if recompute_rl_each_outer:
                                with autocast(device_type="cuda", enabled=use_amp):
                                    exp = per_task_tensors[tid]["exp"]
                                    actions = per_task_tensors[tid]["actions"]
                                    rewards = per_task_tensors[tid]["rewards"]
                                    beh_logits = per_task_tensors[tid]["beh_logits"]
                                    logits_x_all_k, values_x_all_k = policy_net(exp.unsqueeze(0))
                                    cur_logits_x_k = logits_x_all_k[0] if logits_x_all_k.dim() == 3 else logits_x_all_k
                                    values_x_k = values_x_all_k
                                    if values_x_k.dim() == 3:
                                        values_x_k = values_x_k[0, :, 0]
                                    elif values_x_k.dim() == 2:
                                        values_x_k = values_x_k[0]
                                    else:
                                        values_x_k = values_x_k.squeeze()
                                    loss_rl_k, ent_x_k, _ = reinforce_with_baseline(
                                        cur_logits=cur_logits_x_k,
                                        actions=actions,
                                        rewards=rewards,
                                        values=values_x_k,
                                        gamma=args.gamma,
                                        lam=args.gae_lambda,
                                        use_gae=args.use_gae,
                                        entropy_coef=args.explore_entropy_coef,
                                        behavior_logits=beh_logits if args.offpolicy_correction != "none" else None,
                                        offpolicy=args.offpolicy_correction,
                                        is_clip_rho=args.is_clip_rho,
                                    )
                            else:
                                # use k=0 RL once; detach to avoid graph reuse issues
                                loss_rl_k = tensors["loss_rl0"].detach()
                                ent_x_k = float(tensors["ent0"])
                        else:
                            loss_rl_k = torch.tensor(0.0, device=device, dtype=torch.float32)
                            ent_x_k = 0.0
                    else:
                        if recompute_rl_each_outer:
                            with autocast(device_type="cuda", enabled=use_amp):
                                exp = per_task_tensors[tid]["exp"]
                                actions = per_task_tensors[tid]["actions"]
                                rewards = per_task_tensors[tid]["rewards"]
                                beh_logits = per_task_tensors[tid]["beh_logits"]
                                logits_x_all_k, values_x_all_k = policy_net(exp.unsqueeze(0))
                                cur_logits_x_k = logits_x_all_k[0] if logits_x_all_k.dim() == 3 else logits_x_all_k
                                values_x_k = values_x_all_k
                                if values_x_k.dim() == 3:
                                    values_x_k = values_x_k[0, :, 0]
                                elif values_x_k.dim() == 2:
                                    values_x_k = values_x_k[0]
                                else:
                                    values_x_k = values_x_k.squeeze()
                                loss_rl_k, ent_x_k, _ = reinforce_with_baseline(
                                    cur_logits=cur_logits_x_k,
                                    actions=actions,
                                    rewards=rewards,
                                    values=values_x_k,
                                    gamma=args.gamma,
                                    lam=args.gae_lambda,
                                    use_gae=args.use_gae,
                                    entropy_coef=args.explore_entropy_coef,
                                    behavior_logits=beh_logits if args.offpolicy_correction != "none" else None,
                                    offpolicy=args.offpolicy_correction,
                                    is_clip_rho=args.is_clip_rho,
                                )
                        else:
                            # reuse the precomputed value but DETACH so we don't backprop same graph multiple times
                            if k == 0:
                                loss_rl_k = tensors["loss_rl0"]
                            else:
                                loss_rl_k = tensors["loss_rl0"].detach()
                            ent_x_k = float(tensors["ent0"])

                    # --- BC term ---
                    with autocast(device_type="cuda", enabled=use_amp):
                        logits_b, _ = policy_net(batch_obs)  # (B_d, T_max, A)
                        if args.label_smoothing > 0.0:
                            loss_bc_k = smoothed_cross_entropy(
                                logits_b, batch_lab,
                                ignore_index=PAD_ACTION, smoothing=args.label_smoothing
                            )
                        else:
                            ce = nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_k = ce(logits_b.reshape(-1, logits_b.size(-1)), batch_lab.reshape(-1))

                        loss_task = args.rl_coef * loss_rl_k + loss_bc_k

                    total_loss = loss_task if denom == 0 else (total_loss + loss_task)
                    total_bc += float(loss_bc_k.detach().cpu())
                    total_rl += float(loss_rl_k.detach().cpu()) if isinstance(loss_rl_k, torch.Tensor) else float(loss_rl_k)
                    total_ent += float(ent_x_k)
                    denom += 1

                if denom == 0:
                    continue

                avg_loss = total_loss / float(denom)

                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_train_loss += float(avg_loss.detach().cpu())
                running_bc         += (total_bc / denom)
                running_rl         += (total_rl / denom)
                running_ent        += (total_ent / denom)
                count_updates      += 1
                updates_since_free += 1

                # mark reuse for explore refresh gating
                for tid in tasks_used:
                    explore_cache[tid].reuse_count += 1

                if device.type == "cuda" and updates_since_free >= free_every:
                    torch.cuda.empty_cache()
                    updates_since_free = 0

            # --------- free per-batch tensors ----------
            for tid in tasks_used:
                t = per_task_tensors[tid]
                for key in list(t.keys()):
                    t[key] = None
            per_task_tensors.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if count_updates > 0:
                pbar.set_postfix(loss=f"{running_train_loss/max(1,count_updates):.3f}",
                                 bc=f"{running_bc/max(1,count_updates):.3f}",
                                 rl=f"{running_rl/max(1,count_updates):.3f}",
                                 ent=f"{running_ent/max(1,count_updates):.3f}")

        # ------- Validation (rollout) -------
        policy_net.eval()
        with torch.no_grad():
            (val_results, avg_p1, avg_p2, std_p1, std_p2, avg_total, success_rate) = eval_sampled_val(
                policy_net, val_tasks, val_env, device, sample_n=args.val_sample_size
            )

        logger.info(
            f"[Epoch {epoch:02d}] train_loss={running_train_loss/max(1,count_updates):.4f} "
            f"val_phase1={avg_p1:.2f}±{std_p1:.2f} val_phase2={avg_p2:.2f}±{std_p2:.2f} "
            f"success_rate={success_rate:.2f} avg_total_steps={avg_total:.2f}"
        )

        # Model selection on rollout score (lower avg_total is better)
        improved_this = False
        if avg_total < best_val_score:
            best_val_score, best_epoch = avg_total, epoch
            patience, improved, improved_this = 0, True, True
            save_checkpoint(policy_net, epoch, best_val_score, args.save_path, args.load_path)
        else:
            patience += (0 if getattr(args, "disable_early_stop", False) else 1)

        if getattr(args, "disable_early_stop", False) and not improved_this:
            save_checkpoint(policy_net, epoch, best_val_score, args.save_path, args.load_path)

        if (not getattr(args, "disable_early_stop", False)) and patience >= args.early_stop_patience and not improved_this:
            logger.info(f"[EARLY STOP] no improvement for {patience} epochs, stopping.")
            break

    logger.info("BC meta training complete.")

if __name__ == "__main__":
    run_training()
