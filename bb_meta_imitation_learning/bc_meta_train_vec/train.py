# bc_meta_train_vec/train.py

import os
import sys
import json
import math
import random
import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

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

# ---------- env makers ----------
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

def _make_env_fn_with_task(task_cfg: MazeTaskManager.TaskConfig):
    """Thunk that builds a shaping-wrapped env and sets the task before first reset."""
    def _thunk():
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        env = _make_base_env()
        env.unwrapped.set_task(task_cfg)
        env = Phase1ShapingWrapper(env)  # shaping only affects phase-1
        return env
    return _thunk


# ---------- vectorized explore collection ----------
def _collect_explore_vec(
    policy_net,
    task_cfgs: List[MazeTaskManager.TaskConfig],
    device,
    max_steps: int = 250,
    seed_base: Optional[int] = None,
) -> List[ExploreRollout]:
    """
    Collect phase-1 rollouts for a list of tasks using an AsyncVectorEnv.
    Returns a list of ExploreRollout in the same order as task_cfgs.
    Vectorizes BOTH env stepping and policy forward:
      - builds a (B_alive, T_max, 6, H, W) padded batch each tick
      - calls policy_net(...) once per tick, not per env
    Also logs batching sanity metrics.
    """
    import time as _time
    n = len(task_cfgs)
    if n == 0:
        return []

    vec_env = gym.vector.AsyncVectorEnv(
        [_make_env_fn_with_task(cfg) for cfg in task_cfgs],
        shared_memory=False
    )

    # Sanity: confirm vector env type
    try:
        is_async = isinstance(vec_env, gym.vector.AsyncVectorEnv)
    except Exception:
        is_async = "AsyncVectorEnv" in type(vec_env).__name__
    num_envs = getattr(vec_env, "num_envs", n)
    logger.info(f"[VEC] created {type(vec_env).__name__} (async={bool(is_async)}) num_envs={num_envs}")

    # Seed per env if provided
    seeds = None
    if seed_base is not None:
        seeds = [int(seed_base + i) for i in range(n)]

    t0_total = _time.time()
    t_env = 0.0
    t_net = 0.0
    fwd_calls_total = 0
    fwd_calls_batched = 0
    fwd_batch_size_accum = 0

    try:
        # reset (gym API differences)
        try:
            obs_batch, _ = vec_env.reset(seed=seeds)
        except TypeError:
            obs_batch = vec_env.reset(seed=seeds)

        # infer spatial dims
        obs0 = obs_batch[0]
        if obs0.ndim == 3 and obs0.shape[-1] == 3:      # (H, W, 3)
            H, W = int(obs0.shape[0]), int(obs0.shape[1])
        elif obs0.ndim == 3 and obs0.shape[0] == 3:     # (3, H, W)
            H, W = int(obs0.shape[1]), int(obs0.shape[2])
        else:                                           # fallback
            H, W = int(obs0.shape[0]), int(obs0.shape[1])

        # per-env buffers (GPU for speed; moved to CPU when returning)
        states_buf = [torch.empty((max_steps, 6, H, W), device=device, dtype=torch.float32) for _ in range(n)]
        beh_logits_buf: List[List[torch.Tensor]] = [[] for _ in range(n)]
        actions_list: List[List[int]] = [[] for _ in range(n)]
        rewards_list: List[List[float]] = [[] for _ in range(n)]
        steps_i = [0 for _ in range(n)]
        alive = [True for _ in range(n)]
        last_a = [0.0 for _ in range(n)]
        last_r = [0.0 for _ in range(n)]
        last_obs = list(obs_batch)
        n_alive = n

        while n_alive > 0:
            # Build obs6 for all alive envs and write this step into their buffers
            alive_indices = []
            for i in range(n):
                if not alive[i]:
                    continue
                obsi = last_obs[i]
                if obsi.ndim == 3 and obsi.shape[-1] == 3:      # (H, W, 3)
                    img_np = obsi.transpose(2, 0, 1)
                elif obsi.ndim == 3 and obsi.shape[0] == 3:     # (3, H, W)
                    img_np = obsi
                else:
                    img_np = obsi.transpose(2, 0, 1)

                img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32) / 255.0
                pa = torch.full((1, H, W), last_a[i], device=device, dtype=torch.float32)
                pr = torch.full((1, H, W), last_r[i], device=device, dtype=torch.float32)
                bb = torch.zeros((1, H, W), device=device, dtype=torch.float32)  # explore -> 0
                obs6_t = torch.cat([img, pa, pr, bb], dim=0)
                states_buf[i][steps_i[i]] = obs6_t
                alive_indices.append(i)

            # ---- BATCHED POLICY FORWARD over alive envs ----
            t_net_start = _time.time()

            # Build a left-padded batch (B_alive, T_max, 6, H, W)
            B_alive = len(alive_indices)
            T_lens = [steps_i[i] + 1 for i in alive_indices]  # +1 because we just wrote the new step
            T_max = max(T_lens)
            batch_seq = torch.zeros((B_alive, T_max, 6, H, W), device=device, dtype=torch.float32)

            for bi, i in enumerate(alive_indices):
                t = T_lens[bi]
                batch_seq[bi, -t:, :, :, :] = states_buf[i][:t]  # left-pad zeros

            with torch.inference_mode():
                logits_batch, _ = policy_net(batch_seq)  # (B_alive, T_max, A)
                # last valid timestep per env is always index -1 after left-padding
                logits_last = logits_batch[:, -1, :]     # (B_alive, A)

            # sample actions for all alive envs at once
            dist = torch.distributions.Categorical(logits=logits_last)
            actions_alive = dist.sample().detach().cpu().numpy().astype(np.int64)  # (B_alive,)

            # record behavior logits for each env at this step
            for bi, i in enumerate(alive_indices):
                beh_logits_buf[i].append(logits_last[bi].detach().cpu())

            t_net += (_time.time() - t_net_start)
            fwd_calls_total += 1
            fwd_batch_size_accum += B_alive
            if B_alive >= 2:
                fwd_calls_batched += 1

            # Build full action array for vector env step
            actions = np.zeros((n,), dtype=np.int64)
            for bi, i in enumerate(alive_indices):
                actions[i] = actions_alive[bi]

            # ---- Vectorized env step ----
            t_env_start = _time.time()
            obs_batch, rew_batch, term_batch, trunc_batch, info_batch = vec_env.step(actions)
            t_env += (_time.time() - t_env_start)

            # ---- Bookkeeping ----
            for bi, i in enumerate(alive_indices):
                rewards_list[i].append(float(rew_batch[i]))
                actions_list[i].append(int(actions[i]))
                last_a[i] = float(actions[i])
                last_r[i] = float(rew_batch[i])
                last_obs[i] = obs_batch[i]
                steps_i[i] += 1

                if steps_i[i] >= max_steps or bool(term_batch[i]) or bool(trunc_batch[i]):
                    alive[i] = False
                    n_alive -= 1

            # guard: all hit max
            if all(si >= max_steps for si in steps_i):
                break

        # ---- Sanity / timing summary ----
        elapsed = _time.time() - t0_total
        total_steps = sum(steps_i)
        avg_fwd_batch = (fwd_batch_size_accum / max(fwd_calls_total, 1))
        logger.info(
            "[VEC] explore collect: async=%s n_envs=%d total_env_steps=%d "
            "elapsed=%.2fs (env_step=%.2fs, net=%.2fs) throughput=%.1f steps/s steps_per_env=%s "
            "| fwd_calls=%d batched_calls=%d (%.1f%%) avg_fwd_batch=%.2f",
            str(bool(is_async)), n, total_steps,
            elapsed, t_env, t_net,
            (total_steps / max(elapsed, 1e-6)), str(steps_i),
            fwd_calls_total, fwd_calls_batched, 100.0 * (fwd_calls_batched / max(fwd_calls_total, 1)),
            avg_fwd_batch,
        )
        if n > 1 and fwd_calls_batched == 0:
            logger.warning("[VEC][SANITY] No batched policy forwards occurred (B_alive never >= 2). "
                           "This indicates either immediate terminations or a batching bug.")

        # Package results back to CPU
        rollouts: List[ExploreRollout] = []
        for i in range(n):
            T_i = steps_i[i]
            obs6 = states_buf[i][:T_i].contiguous().detach().cpu()
            actions = torch.tensor(actions_list[i], dtype=torch.long)
            rewards = torch.tensor(rewards_list[i], dtype=torch.float32)
            beh_logits = torch.stack(beh_logits_buf[i], dim=0).detach().cpu()
            rollouts.append(
                ExploreRollout(
                    obs6=obs6,
                    actions=actions,
                    rewards=rewards,
                    beh_logits=beh_logits,
                    reuse_count=0,
                )
            )

        # explicit frees
        del states_buf, beh_logits_buf
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return rollouts

    finally:
        try:
            vec_env.close()
        except Exception:
            pass



# ---------- demo concat & mild augmentation ----------
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

def _first_demo_paths(ex_record_list: List[Dict], demo_root: str) -> List[str]:
    demos_p2 = [r for r in ex_record_list if int(r.get("phase", 2)) == 2]
    demos_p2 = sorted(demos_p2, key=lambda r: r["frames"])
    return [os.path.join(demo_root, r["file"]) for r in demos_p2]

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

def _pad_stack_time(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Left-pad a list of (T_i, C, H, W) into a single (B, T_max, C, H, W) tensor with zeros.
    New timesteps are aligned to the right; padding fills the left.
    """
    assert len(tensors) > 0
    device = tensors[0].device
    dtype  = tensors[0].dtype
    C, H, W = tensors[0].shape[1:]
    T_max = max(t.shape[0] for t in tensors)
    B = len(tensors)
    out = torch.zeros((B, T_max, C, H, W), device=device, dtype=dtype)
    for i, t in enumerate(tensors):
        Ti = t.shape[0]
        out[i, -Ti:, ...] = t
    return out


# ---------- training loop ----------
def run_training():
    args = parse_args()
    _setup_logging()
    start_time = datetime.datetime.utcnow()

    # —— New knobs (safe defaults if not present in config) ——
    nbc = int(getattr(args, "nbc", 8))  # number of BC (outer) steps per collected explore
    bc_only_after_first = bool(getattr(args, "bc_only_after_first", False))  # RL only on k==0?
    recompute_rl_each_outer = bool(getattr(args, "recompute_rl_each_outer", True))
    # internal: max envs per vectorized collection batch (no CLI change)
    _vec_cap = int(getattr(args, "num_envs", 8))

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

    # Validation env (single, no shaping)
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

        explore_cache: Dict[int, ExploreRollout] = {}

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

            # ---- Vectorized collection for tasks that need a fresh explore ----
            need_collect = []
            for task in batch_tasks:
                tid = task["task_id"]
                if tid not in explore_cache or explore_cache[tid].reuse_count >= args.explore_reuse_M:
                    need_collect.append(task)
            # collect in chunks to cap the worker pool
            for off in range(0, len(need_collect), max(1, _vec_cap)):
                slice_tasks = need_collect[off: off + max(1, _vec_cap)]
                cfgs = [MazeTaskManager.TaskConfig(**t["task_dict"]) for t in slice_tasks]
                ro_list = _collect_explore_vec(policy_net, cfgs, device, max_steps=250, seed_base=args.seed + 100000 * epoch + 1000 * b + off)
                for t, ro in zip(slice_tasks, ro_list):
                    explore_cache[t["task_id"]] = ro
            
            # =========================
            # NEW: Batched RL precompute on exploration rollouts (k=0)
            # =========================
            exp_list_dev = []
            tids_in_minibatch = []
            for task in batch_tasks:
                tid = task["task_id"]
                if tid not in explore_cache:
                    # extremely defensive: if missing for any reason, collect once
                    cfg = MazeTaskManager.TaskConfig(**task["task_dict"])
                    ro = _collect_explore_vec(
                        policy_net, [cfg], device, max_steps=250,
                        seed_base=args.seed + 424242
                    ).pop()
                    explore_cache[tid] = ro
                exp_list_dev.append(explore_cache[tid].obs6.to(device, non_blocking=True))
                tids_in_minibatch.append(tid)

            if len(exp_list_dev) > 0:
                batch_exp = _pad_stack_time(exp_list_dev)  # (B, T_max, 6, H, W)
                t0_bfwd = time.time()
                with autocast(device_type="cuda", enabled=use_amp):
                    logits_batch_all, values_batch_all = policy_net(batch_exp)  # (B, T_max, A), (B, T_max, *)
                bfwd_elapsed = time.time() - t0_bfwd

                precomp_logits_by_tid: Dict[int, torch.Tensor] = {}
                precomp_values_by_tid: Dict[int, torch.Tensor] = {}

                T_list = [x.shape[0] for x in exp_list_dev]
                for i, tid in enumerate(tids_in_minibatch):
                    Ti = T_list[i]
                    # logits -> (T_i, A)
                    logits_i = logits_batch_all[i, -Ti:, :]
                    # values -> (T_i,)
                    values_i = values_batch_all[i]
                    if values_i.dim() == 3:
                        values_i = values_i[-Ti:, 0]
                    elif values_i.dim() == 2:
                        values_i = values_i[-Ti:, ]
                    else:
                        values_i = values_i.view(-1)[-Ti:]
                    precomp_logits_by_tid[tid] = logits_i
                    precomp_values_by_tid[tid] = values_i

                B = len(exp_list_dev)
                T_max = max(T_list) if T_list else 0
                T_avg = float(sum(T_list) / max(1, len(T_list)))
                logger.info(
                    "[BATCHFWD] RL precompute: B=%d T_max=%d T_avg=%.1f one_fwd=YES time=%.2fs",
                    B, T_max, T_avg, bfwd_elapsed
                )
            else:
                precomp_logits_by_tid = {}
                precomp_values_by_tid = {}

            # ---- per-task optimization (unchanged) ----
            for task in batch_tasks:
                tid = task["task_id"]
                cfg = MazeTaskManager.TaskConfig(**task["task_dict"])

                # KL-based refresh against current policy if cache exists
                did_refresh_kl = False
                if tid in explore_cache:
                    with torch.no_grad():
                        seq_dev_tmp = explore_cache[tid].obs6.to(device, non_blocking=True).unsqueeze(0)
                        logits_all_tmp, _ = policy_net(seq_dev_tmp)  # (1,Tx,A)
                        cur_logits_tmp = logits_all_tmp[0] if logits_all_tmp.dim() == 3 else logits_all_tmp
                        if mean_kl_logits(cur_logits_tmp, explore_cache[tid].beh_logits.to(device)).item() > args.kl_refresh_threshold:
                            # recollect just this one (vec size = 1)
                            ro = _collect_explore_vec(policy_net, [cfg], device, max_steps=250, seed_base=args.seed + 999999).pop()
                            explore_cache[tid] = ro
                            did_refresh_kl = True
                    try:
                        del seq_dev_tmp, logits_all_tmp, cur_logits_tmp
                    except NameError:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    # if somehow still missing, collect once
                    ro = _collect_explore_vec(policy_net, [cfg], device, max_steps=250, seed_base=args.seed + 424242).pop()
                    explore_cache[tid] = ro

                # If KL refresh changed the rollout, drop any stale precompute for this tid
                if did_refresh_kl:
                    if tid in precomp_logits_by_tid:
                        del precomp_logits_by_tid[tid]
                    if tid in precomp_values_by_tid:
                        del precomp_values_by_tid[tid]

                # -------- Copy explore chunk to device --------
                exp_six_dev  = explore_cache[tid].obs6.to(device, non_blocking=True)      # (Tx,6,H,W)
                actions_x    = explore_cache[tid].actions.to(device, non_blocking=True)   # (Tx,)
                rewards_x    = explore_cache[tid].rewards.to(device, non_blocking=True)   # (Tx,)
                beh_logits_x = explore_cache[tid].beh_logits.to(device, non_blocking=True)# (Tx,A)
                Tx = exp_six_dev.shape[0]

                # ensure optional forward-result names exist (so cleanup never errors)
                logits_x_all = None
                values_x_all = None
                cur_logits_x0 = None
                values_x0 = None

                # -------- Precompute RL once (k=0), optional recompute later --------
                with autocast(device_type="cuda", enabled=use_amp):
                    use_precomp = False
                    if (tid in precomp_logits_by_tid) and (tid in precomp_values_by_tid):
                        cur_logits_x0 = precomp_logits_by_tid[tid]
                        values_x0     = precomp_values_by_tid[tid]
                        # Shape guard: only use if lengths match current Tx
                        if cur_logits_x0.size(0) == Tx and values_x0.size(0) == Tx:
                            use_precomp = True
                    if not use_precomp:
                        logits_x_all, values_x_all = policy_net(exp_six_dev.unsqueeze(0))
                        cur_logits_x0 = logits_x_all[0] if logits_x_all.dim() == 3 else logits_x_all
                        values_x0 = values_x_all
                        if values_x0.dim() == 3:
                            values_x0 = values_x0[0, :, 0]
                        elif values_x0.dim() == 2:
                            values_x0 = values_x0[0]
                        else:
                            values_x0 = values_x0.squeeze()
                        # Normalize shapes
                        cur_logits_x0 = cur_logits_x0[-Tx:, :]  # (Tx, A)
                        values_x0 = values_x0.view(-1)[-Tx:]    # (Tx,)

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

                # ESS refresh (based on taken actions)
                with torch.no_grad():
                    lp_cur = torch.log_softmax(cur_logits_x0, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                    lp_beh = torch.log_softmax(beh_logits_x, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                    rhos = torch.exp(lp_cur - lp_beh)
                    ess_ratio = ess_ratio_from_rhos(rhos).item()
                if args.explore_reuse_M > 1 and ess_ratio < args.ess_refresh_ratio:
                    ro = _collect_explore_vec(policy_net, [cfg], device, max_steps=250, seed_base=args.seed + 31337).pop()
                    explore_cache[tid] = ro
                    exp_six_dev  = ro.obs6.to(device, non_blocking=True)
                    actions_x    = ro.actions.to(device, non_blocking=True)
                    rewards_x    = ro.rewards.to(device, non_blocking=True)
                    beh_logits_x = ro.beh_logits.to(device, non_blocking=True)
                    Tx = exp_six_dev.shape[0]
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
                        # Normalize shapes after refresh
                        cur_logits_x0 = cur_logits_x0[-Tx:, :]
                        values_x0 = values_x0.view(-1)[-Tx:]

                # -------- Build batched BC tensors from ALL demos (once) --------
                prev_action_start = float(actions_x[-1].item()) if Tx > 0 else 0.0

                demo_obs_list: List[torch.Tensor] = []
                demo_lab_list: List[torch.Tensor] = []

                for demo_path in task["p2_paths"]:
                    p2_six, p2_labels = _load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start)
                    if p2_six.numel() == 0:
                        continue
                    p2_six = _maybe_augment_demo_six_cpu(p2_six, args)  # augmentation (CPU)
                    p2_six = p2_six.to(device, non_blocking=True)
                    p2_labels = p2_labels.to(device, non_blocking=True)
                    obs6_cat, labels_cat, _, _ = _concat_explore_and_exploit(exp_six_dev, p2_six, p2_labels)
                    demo_obs_list.append(obs6_cat)
                    demo_lab_list.append(labels_cat)

                if len(demo_obs_list) == 0:
                    # free explore dev copies before continue (safe None assignments)
                    exp_six_dev = None
                    actions_x = None
                    rewards_x = None
                    beh_logits_x = None
                    logits_x_all = None
                    values_x_all = None
                    cur_logits_x0 = None
                    values_x0 = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
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

                # ---- MULTIPLE OUTER (BC) STEPS reusing the SAME explore ----
                for k in range(nbc):
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(device_type="cuda", enabled=use_amp):
                        # Decide RL contribution this step
                        if k == 0 and bc_only_after_first:
                            loss_rl_k, ent_x_k = loss_rl_0, float(0.0) if isinstance(loss_rl_0, torch.Tensor) else 0.0
                            ent_x_k = ent_x_k if isinstance(ent_x_k, float) else float(ent_x_k)
                        elif not bc_only_after_first and not recompute_rl_each_outer:
                            loss_rl_k, ent_x_k = loss_rl_0, float(0.0)
                        elif recompute_rl_each_outer:
                            logits_x_all_k, values_x_all_k = policy_net(exp_six_dev.unsqueeze(0))
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
                                actions=actions_x,
                                rewards=rewards_x,
                                values=values_x_k,
                                gamma=args.gamma,
                                lam=args.gae_lambda,
                                use_gae=args.use_gae,
                                entropy_coef=args.explore_entropy_coef,
                                behavior_logits=beh_logits_x if args.offpolicy_correction != "none" else None,
                                offpolicy=args.offpolicy_correction,
                                is_clip_rho=args.is_clip_rho,
                            )
                        else:
                            # BC-only step (k>0)
                            loss_rl_k = torch.tensor(0.0, device=device, dtype=torch.float32)
                            ent_x_k = 0.0

                        # BC on demos (reuse same batch each outer step)
                        logits_b, _ = policy_net(batch_obs)  # (B_d, T_max, A)
                        if args.label_smoothing > 0.0:
                            loss_bc_k = smoothed_cross_entropy(
                                logits_b, batch_lab,
                                ignore_index=PAD_ACTION, smoothing=args.label_smoothing
                            )
                        else:
                            ce = nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_k = ce(logits_b.reshape(-1, logits_b.size(-1)), batch_lab.reshape(-1))

                        loss_k = args.rl_coef * loss_rl_k + loss_bc_k

                    scaler.scale(loss_k).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    # update logs
                    running_train_loss += float(loss_k.detach().cpu())
                    running_bc         += float(loss_bc_k.detach().cpu())
                    running_rl         += float(loss_rl_k.detach().cpu())
                    running_ent        += float(ent_x_k)
                    count_updates      += 1
                    updates_since_free += 1

                    # count reuse of explore (so refresh gates work)
                    explore_cache[tid].reuse_count += 1

                    if device.type == "cuda" and updates_since_free >= free_every:
                        torch.cuda.empty_cache()
                        updates_since_free = 0

                # ---- free big tensors ASAP (safe None assignments) ----
                exp_six_dev = None
                actions_x = None
                rewards_x = None
                beh_logits_x = None
                logits_x_all = None
                values_x_all = None
                cur_logits_x0 = None
                values_x0 = None
                demo_obs_list = None
                demo_lab_list = None
                pad_obs = None
                pad_lab = None
                batch_obs = None
                batch_lab = None
                logits_b = None
                loss_bc_k = None
                loss_k = None
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

    logger.info("BC meta training (vectorized explore) complete.")

if __name__ == "__main__":
    run_training()
