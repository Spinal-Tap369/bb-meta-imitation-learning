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

# --- functional call (PyTorch 2.0+) with fallback to functorch ---
try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    try:
        from functorch import functional_call  # type: ignore
    except Exception:
        functional_call = None  # will raise later if actually used

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

# [DBG] ---- debug helpers ----
def _maybe_set_debug_level(debug: bool, level: str):
    if not debug:
        return
    level = level.upper().strip()
    lvl = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}.get(level, logging.INFO)
    logging.getLogger().setLevel(lvl)
    logging.getLogger(__name__).setLevel(lvl)

def _cuda_mem(prefix: str, device):
    if device.type != "cuda":
        return "CPU"
    a = torch.cuda.memory_allocated() / (1024**2)
    r = torch.cuda.memory_reserved() / (1024**2)
    return f"{prefix} mem: alloc={a:.1f}MB reserved={r:.1f}MB"

def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t.detach()).all().item() if isinstance(t, torch.Tensor) else True

def _shape_str(t):
    if isinstance(t, torch.Tensor):
        return f"{tuple(t.shape)} {str(t.dtype).replace('torch.','')}"
    return str(type(t))

def _count_params(mod: nn.Module):
    tot = sum(p.numel() for p in mod.parameters())
    train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return tot, train

def _timer():
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0)

# ---------- small functional utils for MRI-style inner step ----------
def _named_params_and_buffers(net: nn.Module):
    """Return separate dicts of named parameters and buffers."""
    p = dict(net.named_parameters())
    b = dict(net.named_buffers())
    return p, b

def _merge_params_and_buffers(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor]):
    """Functional call expects a single mapping of {name: tensor} for both."""
    merged = {}
    merged.update(buffers)  # buffers first so params override in unlikely name collision
    merged.update(params)
    return merged

def _split_params_head_vs_rest(
    params: Dict[str, torch.Tensor],
    head_name_substr=("policy_head", "action_head", "logits")
):
    """Return (head_params, rest_params) by name substring match."""
    head = {k: v for k, v in params.items() if any(s in k for s in head_name_substr)}
    rest = {k: v for k, v in params.items() if k not in head}
    return head, rest

def _functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], x: torch.Tensor):
    if functional_call is None:
        raise RuntimeError("functional_call is unavailable. Please use PyTorch >= 2.0 or install functorch.")
    mapping = _merge_params_and_buffers(params, buffers)
    return functional_call(model, mapping, (x,))

def _split_trainable(params: Dict[str, torch.Tensor]):
    trainable = {k: v for k, v in params.items() if v.requires_grad}
    frozen    = {k: v for k, v in params.items() if not v.requires_grad}
    return trainable, frozen


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
    # [DBG] flags forwarded from outer scope for ad-hoc profiling/logging:
    dbg=False, dbg_timing=True, dbg_level="INFO"
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

    # [DBG] env sanity
    if dbg:
        try:
            is_async = isinstance(vec_env, gym.vector.AsyncVectorEnv)
        except Exception:
            is_async = "AsyncVectorEnv" in type(vec_env).__name__
        logger.log(logging.DEBUG if dbg_level=="DEBUG" else logging.INFO,
                   f"[VEC] created {type(vec_env).__name__} async={bool(is_async)} num_envs={getattr(vec_env,'num_envs',n)}")

    # [DBG] timers
    t0_total = _time.time()
    t_env = 0.0
    t_net = 0.0
    fwd_calls_total = 0
    fwd_calls_batched = 0
    fwd_batch_size_accum = 0

    try:
        try:
            obs_batch, _ = vec_env.reset(seed=([int(seed_base + i) for i in range(n)] if seed_base is not None else None))
        except TypeError:
            obs_batch = vec_env.reset(seed=([int(seed_base + i) for i in range(n)] if seed_base is not None else None))

        obs0 = obs_batch[0]
        if obs0.ndim == 3 and obs0.shape[-1] == 3:      # (H, W, 3)
            H, W = int(obs0.shape[0]), int(obs0.shape[1])
        elif obs0.ndim == 3 and obs0.shape[0] == 3:     # (3, H, W)
            H, W = int(obs0.shape[1]), int(obs0.shape[2])
        else:                                           # fallback
            H, W = int(obs0.shape[0]), int(obs0.shape[1])

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
            alive_indices = []
            for i in range(n):
                if not alive[i]:
                    continue
                obsi = last_obs[i]
                if obsi.ndim == 3 and obsi.shape[-1] == 3:
                    img_np = obsi.transpose(2, 0, 1)
                elif obsi.ndim == 3 and obsi.shape[0] == 3:
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

            # batched policy forward over alive envs
            t_net_start = _time.time()
            B_alive = len(alive_indices)
            T_lens = [steps_i[i] + 1 for i in alive_indices]
            T_max = max(T_lens)
            batch_seq = torch.zeros((B_alive, T_max, 6, H, W), device=device, dtype=torch.float32)
            for bi, i in enumerate(alive_indices):
                t = T_lens[bi]
                batch_seq[bi, -t:, :, :, :] = states_buf[i][:t]

            with torch.inference_mode():
                logits_batch, _ = policy_net(batch_seq)  # (B_alive, T_max, A)
                logits_last = logits_batch[:, -1, :]     # (B_alive, A)

            dist = torch.distributions.Categorical(logits=logits_last)
            actions_alive = dist.sample().detach().cpu().numpy().astype(np.int64)

            for bi, i in enumerate(alive_indices):
                beh_logits_buf[i].append(logits_last[bi].detach().cpu())

            t_net += (_time.time() - t_net_start)
            fwd_calls_total += 1
            fwd_batch_size_accum += B_alive
            if B_alive >= 2:
                fwd_calls_batched += 1

            # vectorized env step
            t_env_start = _time.time()
            actions = np.zeros((n,), dtype=np.int64)
            for bi, i in enumerate(alive_indices):
                actions[i] = actions_alive[bi]
            obs_batch, rew_batch, term_batch, trunc_batch, info_batch = vec_env.step(actions)
            t_env += (_time.time() - t_env_start)

            # bookkeeping
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

            if all(si >= max_steps for si in steps_i):
                break

        # [DBG] summary
        if dbg:
            elapsed = _time.time() - t0_total
            total_steps = sum(steps_i)
            avg_fwd_batch = (fwd_batch_size_accum / max(fwd_calls_total, 1))
            logger.info(
                "[VEC] explore collect done: envs=%d steps_total=%d elapsed=%.2fs env=%.2fs net=%.2fs "
                "throughput=%.1f steps/s fwd_calls=%d batched_calls=%d (%.1f%%) avg_fwd_batch=%.2f",
                n, total_steps, elapsed, t_env, t_net,
                (total_steps / max(elapsed, 1e-6)), fwd_calls_total, fwd_calls_batched,
                100.0 * (fwd_calls_batched / max(fwd_calls_total, 1)), avg_fwd_batch
            )
            if n > 1 and fwd_calls_batched == 0:
                logger.warning("[VEC][SANITY] No batched policy forwards occurred (B_alive never >= 2).")

        # package to CPU
        rollouts: List[ExploreRollout] = []
        for i in range(n):
            T_i = steps_i[i]
            obs6 = states_buf[i][:T_i].contiguous().detach().cpu()
            actions = torch.tensor(actions_list[i], dtype=torch.long)
            rewards = torch.tensor(rewards_list[i], dtype=torch.float32)
            beh_logits = torch.stack(beh_logits_buf[i], dim=0).detach().cpu()
            rollouts.append(ExploreRollout(obs6=obs6, actions=actions, rewards=rewards, beh_logits=beh_logits, reuse_count=0))

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
    nbc = int(getattr(args, "nbc", 16))  # number of BC (outer) steps per collected explore
    # MRI extras (not in config.py, use getattr defaults):
    inner_lr = float(getattr(args, "inner_lr", 1e-3))
    inner_trunc_T = getattr(args, "inner_trunc_T", None)  # e.g., 128 to truncate explore for inner loss
    head_only_inner = bool(getattr(args, "head_only_inner", False))

    # [DBG] logging flags (non-breaking)
    debug = bool(getattr(args, "debug", True))
    debug_level = str(getattr(args, "debug_level", "INFO")).upper()
    debug_every_batches = int(getattr(args, "debug_every_batches", 1))
    debug_tasks_per_batch = int(getattr(args, "debug_tasks_per_batch", 4))
    debug_inner_per_task = bool(getattr(args, "debug_inner_per_task", False))
    debug_mem = bool(getattr(args, "debug_mem", True))
    debug_timing = bool(getattr(args, "debug_timing", True))
    debug_shapes = bool(getattr(args, "debug_shapes", True))
    _maybe_set_debug_level(debug, debug_level)

    logger.info("[SETUP] debug=%s level=%s nbc=%d inner_lr=%.2e inner_trunc_T=%s head_only_inner=%s",
                str(debug), debug_level, nbc, inner_lr, str(inner_trunc_T), str(head_only_inner))

    # Seeding
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("[SETUP] seed=%d cudnn.deterministic=%s benchmark=%s", args.seed, True, False)

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
    logger.info("[SPLIT] train=%d val=%d (val_size=%d)", len(train_tasks), len(val_tasks), n_val)

    # Model
    policy_net = build_model(seq_len=SEQ_LEN).to(device)
    tot, trn = _count_params(policy_net)
    logger.info("[MODEL] built model params total=%d trainable=%d", tot, trn)

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

    if debug_shapes:
        logger.info("[SHAPES] SEQ_LEN=%s", str(SEQ_LEN))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        final_epoch = epoch
        logger.info("========== [EPOCH %02d/%02d] ==========", epoch, args.epochs)

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
        logger.info("[TRAIN] num_batches=%d batch_size=%d", num_batches, args.batch_size)

        policy_net.train()
        running_bc = 0.0
        running_inrl = 0.0  # monitor inner RL loss at θ (detached)
        count_updates = 0

        pbar = tqdm(range(num_batches), desc=f"[Epoch {epoch:02d}] train", leave=False)
        for b in pbar:
            is_verbose_batch = debug and ((b % max(1, debug_every_batches)) == 0)
            btimer = _timer() if debug_timing else None

            start = b * args.batch_size
            end = min(len(train_tasks), (b + 1) * args.batch_size)
            batch_ids = [batch_indices[i] for i in range(start, end)]
            batch_tasks = [train_tasks[i] for i in batch_ids]
            if is_verbose_batch:
                tids = [t["task_id"] for t in batch_tasks]
                logger.info("[BATCH %d/%d] tasks=%s", b+1, num_batches, tids)

            # ---- Vectorized collection for tasks that need a fresh explore ----
            need_collect = []
            for task in batch_tasks:
                tid = task["task_id"]
                if tid not in explore_cache or explore_cache[tid].reuse_count >= args.explore_reuse_M:
                    need_collect.append(task)
            _vec_cap = int(getattr(args, "num_envs", 8))
            if is_verbose_batch:
                logger.info("[COLLECT] need_collect=%d vec_cap=%d", len(need_collect), _vec_cap)

            for off in range(0, len(need_collect), max(1, _vec_cap)):
                slice_tasks = need_collect[off: off + max(1, _vec_cap)]
                cfgs = [MazeTaskManager.TaskConfig(**t["task_dict"]) for t in slice_tasks]
                ro_list = _collect_explore_vec(
                    policy_net, cfgs, device, max_steps=250,
                    seed_base=args.seed + 100000 * epoch + 1000 * b + off,
                    dbg=is_verbose_batch, dbg_timing=debug_timing, dbg_level=debug_level
                )
                for t, ro in zip(slice_tasks, ro_list):
                    explore_cache[t["task_id"]] = ro
                    if is_verbose_batch and debug_shapes:
                        logger.info("[COLLECT] tid=%s Tx=%d obs6=%s actions=%s rewards=%s",
                                    t["task_id"], ro.obs6.shape[0], _shape_str(ro.obs6),
                                    _shape_str(ro.actions), _shape_str(ro.rewards))

            # =========================
            # BUILD per-task tensors ONCE (used across k outer steps)
            # =========================
            per_task_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
            tasks_used: List[int] = []

            for idx_task, task in enumerate(batch_tasks):
                tid = task["task_id"]
                cfg = MazeTaskManager.TaskConfig(**task["task_dict"])

                did_refresh_kl = False
                if tid in explore_cache:
                    with torch.no_grad():
                        seq_dev_tmp = explore_cache[tid].obs6.to(device, non_blocking=True).unsqueeze(0)
                        logits_all_tmp, _ = policy_net(seq_dev_tmp)  # (1,Tx,A)
                        cur_logits_tmp = logits_all_tmp[0] if logits_all_tmp.dim() == 3 else logits_all_tmp
                        kl = mean_kl_logits(cur_logits_tmp, explore_cache[tid].beh_logits.to(device)).item()
                        if is_verbose_batch and idx_task < debug_tasks_per_batch:
                            logger.info("[KL] tid=%s mean_kl=%.4f thr=%.4f", tid, kl, args.kl_refresh_threshold)
                        if kl > args.kl_refresh_threshold:
                            ro = _collect_explore_vec(policy_net, [cfg], device, max_steps=250,
                                                      seed_base=args.seed + 999999,
                                                      dbg=is_verbose_batch, dbg_timing=debug_timing, dbg_level=debug_level).pop()
                            explore_cache[tid] = ro
                            did_refresh_kl = True
                    try:
                        del seq_dev_tmp, logits_all_tmp, cur_logits_tmp
                    except NameError:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    ro = _collect_explore_vec(policy_net, [cfg], device, max_steps=250,
                                              seed_base=args.seed + 424242,
                                              dbg=is_verbose_batch, dbg_timing=debug_timing, dbg_level=debug_level).pop()
                    explore_cache[tid] = ro

                # -------- Copy explore chunk to device --------
                ro = explore_cache[tid]
                exp_six_dev  = ro.obs6.to(device, non_blocking=True)      # (Tx,6,H,W)
                actions_x    = ro.actions.to(device, non_blocking=True)   # (Tx,)
                rewards_x    = ro.rewards.to(device, non_blocking=True)   # (Tx,)
                beh_logits_x = ro.beh_logits.to(device, non_blocking=True)# (Tx,A)
                Tx = exp_six_dev.shape[0]

                # ESS refresh (based on taken actions) using current θ (no grad)
                with torch.no_grad():
                    logits_now, _ = policy_net(exp_six_dev.unsqueeze(0))
                    logits_now = logits_now[0] if logits_now.dim() == 3 else logits_now  # (Tx, A)
                    lp_cur = torch.log_softmax(logits_now, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                    lp_beh = torch.log_softmax(beh_logits_x, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                    rhos = torch.exp(lp_cur - lp_beh)
                    ess_ratio = ess_ratio_from_rhos(rhos).item()
                if is_verbose_batch and idx_task < debug_tasks_per_batch:
                    logger.info("[ESS] tid=%s Tx=%d ess_ratio=%.3f thr=%.3f reuse_count=%d",
                                tid, Tx, ess_ratio, args.ess_refresh_ratio, ro.reuse_count)
                if int(getattr(args, "explore_reuse_M", 1)) > 1 and ess_ratio < args.ess_refresh_ratio:
                    ro = _collect_explore_vec(policy_net, [cfg], device, max_steps=250,
                                              seed_base=args.seed + 31337,
                                              dbg=is_verbose_batch, dbg_timing=debug_timing, dbg_level=debug_level).pop()
                    explore_cache[tid] = ro
                    exp_six_dev  = ro.obs6.to(device, non_blocking=True)
                    actions_x    = ro.actions.to(device, non_blocking=True)
                    rewards_x    = ro.rewards.to(device, non_blocking=True)
                    beh_logits_x = ro.beh_logits.to(device, non_blocking=True)
                    Tx = exp_six_dev.shape[0]
                    if is_verbose_batch and idx_task < debug_tasks_per_batch:
                        logger.info("[ESS][REFRESH] tid=%s new_Tx=%d", tid, Tx)

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
                    if is_verbose_batch and idx_task < debug_tasks_per_batch:
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

                if is_verbose_batch and idx_task < debug_tasks_per_batch and debug_shapes:
                    logger.info("[BC][BUILD] tid=%s B_d=%d T_max=%d obs=%s lab=%s",
                                tid, B_d, T_max, _shape_str(batch_obs), _shape_str(batch_lab))

                per_task_tensors[tid] = {
                    "batch_obs": batch_obs,       # for BC outer at φ
                    "batch_lab": batch_lab,
                    "exp": exp_six_dev,           # for inner RL step (θ -> φ)
                    "actions": actions_x,
                    "rewards": rewards_x,
                    "beh_logits": beh_logits_x,
                }
                tasks_used.append(tid)

            # =========================
            # OUTER LOOP: MRI-style (k steps over the whole minibatch)
            # =========================
            if len(tasks_used) == 0:
                continue

            if is_verbose_batch and debug_mem:
                logger.info("[MEM][PRE-K] %s", _cuda_mem("pre-k", device))

            for k in range(nbc):
                ktimer = _timer() if debug_timing else None
                params_theta_named, buffers_named = _named_params_and_buffers(policy_net)
                # Split trainable vs frozen (handles encoder warmup safely)
                trainable_named, frozen_named = _split_trainable(params_theta_named)
                if head_only_inner:
                    head_theta_named, _ = _split_params_head_vs_rest(params_theta_named)
                else:
                    head_theta_named = trainable_named
                
                # Everything not updated in the inner step goes to "rest" (include frozen!)
                rest_theta_named = {**frozen_named, **{k: v for k, v in params_theta_named.items() if k not in head_theta_named}}

                # Debug logging
                if bool(getattr(args, "debug", True)):
                    logger.info(
                        f"[INNER] k-loop={nbc} head_only={head_only_inner} "
                        f"head_params={len(head_theta_named)} rest_params={len(rest_theta_named)} "
                        f"trainable={len(trainable_named)} frozen={len(frozen_named)}"
                    )

                if is_verbose_batch and k == 0:
                    logger.info("[INNER] k-loop=%d head_only=%s head_params=%d rest_params=%d",
                                nbc, str(head_only_inner), len(head_theta_named), len(rest_theta_named))

                loss_bc_list = []
                inrl_monitor_sum = 0.0

                optimizer.zero_grad(set_to_none=True)

                for idx_task, tid in enumerate(tasks_used):
                    tensors = per_task_tensors[tid]
                    exp = tensors["exp"]
                    acts = tensors["actions"]
                    rews = tensors["rewards"]
                    beh  = tensors["beh_logits"]

                    Tx_full = exp.shape[0]
                    if inner_trunc_T is not None and isinstance(inner_trunc_T, int) and exp.shape[0] > inner_trunc_T:
                        exp = exp[-inner_trunc_T:]
                        acts = acts[-inner_trunc_T:]
                        rews = rews[-inner_trunc_T:]
                        beh  = beh[-inner_trunc_T:]
                    if is_verbose_batch and idx_task < debug_tasks_per_batch and k == 0:
                        logger.info("[INNER][TRUNC] tid=%s Tx_full=%d Tx_used=%d", tid, Tx_full, exp.shape[0])

                    # (a) Inner RL loss at θ
                    with autocast(device_type="cuda", enabled=use_amp):
                        logits_all, values_all = _functional_forward(
                            policy_net,
                            {**rest_theta_named, **head_theta_named},
                            buffers_named,
                            exp.unsqueeze(0)  # (1,Tx,6,H,W)
                        )
                        logits_x = logits_all[0]          # (Tx, A)
                        values_x = values_all[0]          # (Tx,)

                        loss_in, ent_x, _ = reinforce_with_baseline(
                            cur_logits=logits_x,
                            actions=acts,
                            rewards=rews,
                            values=values_x,
                            gamma=args.gamma, lam=args.gae_lambda, use_gae=args.use_gae,
                            entropy_coef=args.explore_entropy_coef,
                            behavior_logits=(beh if args.offpolicy_correction!="none" else None),
                            offpolicy=args.offpolicy_correction,
                            is_clip_rho=args.is_clip_rho,
                        )

                    inrl_monitor_sum += float(loss_in.detach().cpu())
                    if is_verbose_batch and debug_inner_per_task and idx_task < debug_tasks_per_batch:
                        logger.info("[INNER][LOSS] k=%d tid=%s inrl=%.4f ent=%.4f", k, tid, float(loss_in), float(ent_x))

                    # (b) One differentiable inner step: θ -> φ on head params
                    grads = torch.autograd.grad(
                        loss_in, tuple(head_theta_named.values()), create_graph=True, allow_unused=True
                    )
                    if is_verbose_batch and idx_task == 0 and k == 0:
                        n_nan = sum((not _finite(g)) for g in grads if isinstance(g, torch.Tensor))
                        if n_nan > 0:
                            logger.warning("[GRAD][NONFINITE] inner grads contain %d non-finite entries", n_nan)
                    phi_head = {}
                    for (k, w), g in zip(head_theta_named.items(), grads):
                        if g is None:
                            if bool(getattr(args, "debug", True)):
                                logger.warning(f"[INNER][GRAD] None grad for '{k}'; using zeros")
                            g = torch.zeros_like(w)
                        phi_head[k] = w - inner_lr * g

                    # (c) Evaluate BC at φ
                    with autocast(device_type="cuda", enabled=use_amp):
                        phi_params = {**rest_theta_named, **phi_head}
                        logits_b, _ = _functional_forward(policy_net, phi_params, buffers_named, tensors["batch_obs"])
                        if args.label_smoothing > 0.0:
                            loss_bc_k = smoothed_cross_entropy(
                                logits_b, tensors["batch_lab"], ignore_index=PAD_ACTION, smoothing=args.label_smoothing
                            )
                        else:
                            ce = nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                            loss_bc_k = ce(logits_b.reshape(-1, logits_b.size(-1)), tensors["batch_lab"].reshape(-1))

                    loss_bc_list.append(loss_bc_k)

                if len(loss_bc_list) == 0:
                    continue

                total_bc = sum(loss_bc_list) / float(len(loss_bc_list))
                avg_loss = total_bc

                # Backprop
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                # capture grad norm
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    logger.warning("[GRAD][NONFINITE] grad_norm=%s — skipping optimizer step", str(grad_norm))
                    optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.step(optimizer)
                    scaler.update()

                # reuse count
                for tid in tasks_used:
                    explore_cache[tid].reuse_count += 1

                # stats
                running_bc += float(total_bc.detach().cpu())
                running_inrl += inrl_monitor_sum / max(1, len(tasks_used))
                count_updates += 1

                if device.type == "cuda":
                    updates_since_free += 1
                    if updates_since_free >= free_every:
                        torch.cuda.empty_cache()
                        updates_since_free = 0

                if is_verbose_batch:
                    msg = f"[STEP] k={k+1}/{nbc} avg_bc={float(total_bc):.4f} inrl@theta(avg)={inrl_monitor_sum/max(1,len(tasks_used)):.4f} grad_norm={float(grad_norm):.3f}"
                    if debug_mem:
                        msg += " | " + _cuda_mem("post-step", device)
                    if debug_timing and ktimer is not None:
                        msg += f" | step_time={ktimer():.3f}s"
                    logger.info(msg)

            # --------- free per-batch tensors ----------
            for tid in tasks_used:
                tdict = per_task_tensors[tid]
                for key in list(tdict.keys()):
                    tdict[key] = None
            per_task_tensors.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if count_updates > 0:
                pbar.set_postfix(
                    bc=f"{running_bc/max(1,count_updates):.3f}",
                    inrl=f"{running_inrl/max(1,count_updates):.3f}"
                )

            if is_verbose_batch and debug_timing and btimer is not None:
                logger.info("[BATCH %d] elapsed=%.3fs", b+1, btimer())

        # ------- Validation (rollout) -------
        policy_net.eval()
        with torch.no_grad():
            vtimer = _timer() if debug_timing else None
            (val_results, avg_p1, avg_p2, std_p1, std_p2, avg_total, success_rate) = eval_sampled_val(
                policy_net, val_tasks, val_env, device, sample_n=args.val_sample_size
            )
            if debug and debug_timing and vtimer is not None:
                logger.info("[VAL] elapsed=%.3fs", vtimer())

        logger.info(
            f"[Epoch {epoch:02d}] bc_outer={running_bc/max(1,count_updates):.4f} "
            f"inrl@theta={running_inrl/max(1,count_updates):.4f} "
            f"val_phase1={avg_p1:.2f}±{std_p1:.2f} val_phase2={avg_p2:.2f}±{std_p2:.2f} "
            f"success_rate={success_rate:.2f} avg_total_steps={avg_total:.2f}"
        )

        # Model selection on rollout score (lower avg_total is better)
        improved_this = False
        if avg_total < best_val_score:
            best_val_score, best_epoch = avg_total, epoch
            patience, improved, improved_this = 0, True, True
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

    logger.info("BC meta training (MRI-style inner step; vectorized explore) complete.")

if __name__ == "__main__":
    run_training()
