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
from torch.nn import CrossEntropyLoss
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
)
from .rl_loss import reinforce_with_baseline, mean_kl_logits, ess_ratio_from_rhos

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
    obs6: torch.Tensor           # (T,6,H,W) in [0,1]  (kept on device)
    actions: torch.Tensor        # (T,)                (device)
    rewards: torch.Tensor        # (T,)                (device)
    beh_logits: torch.Tensor     # (T,A) logits at collection time (device)
    reuse_count: int             # how many updates we've used this for (this epoch)

def _collect_explore(
    policy_net,
    env,
    task_cfg,
    device,
    max_steps: int = 250,
) -> ExploreRollout:
    """Run phase-1 on-policy in the env and return tensors for RL training.
       Optimized: pre-allocate GPU buffer; no per-step np.stack; keep outputs on device."""
    env.unwrapped.set_task(task_cfg)
    try:
        env.unwrapped.maze_core.randomize_start()
        env.unwrapped.maze_core.randomize_goal(min_distance=3.0)
    except Exception:
        pass

    obs, _ = env.reset()
    done = False
    trunc = False

    # infer spatial size from first obs
    H, W = obs.shape[0], obs.shape[1]

    # preallocate on GPU
    states_buf = torch.empty((max_steps, 6, H, W), device=device, dtype=torch.float32)
    beh_logits_buf: List[torch.Tensor] = []
    actions_list: List[int] = []
    rewards_list: List[float] = []

    last_a, last_r = 0.0, 0.0
    steps = 0
    reached = False

    with torch.inference_mode():
        while not done and not trunc and steps < max_steps:
            # build obs6 on GPU
            img = torch.from_numpy(obs.transpose(2, 0, 1)).to(device=device, dtype=torch.float32) / 255.0
            pa = torch.full((1, H, W), last_a, device=device, dtype=torch.float32)
            pr = torch.full((1, H, W), last_r, device=device, dtype=torch.float32)
            bb = torch.zeros((1, H, W), device=device, dtype=torch.float32)  # explore -> 0
            obs6_t = torch.cat([img, pa, pr, bb], dim=0)
            states_buf[steps] = obs6_t

            # policy step on the prefix [:steps+1]
            seq = states_buf[: steps + 1].unsqueeze(0)  # (1, t, 6,H,W)
            logits, _ = policy_net.act_single_step(seq)  # logits: (1,A) or (A,)
            logits_t = logits[0] if logits.dim() == 2 else logits
            beh_logits_buf.append(logits_t.detach())

            action = torch.distributions.Categorical(logits=logits_t).sample().item()
            obs, rew, done, trunc, info = env.step(action)
            if env.unwrapped.maze_core.phase_metrics[2]["goal_rewards"] > 0:
                reached = True

            actions_list.append(action)
            rewards_list.append(float(rew))
            last_a, last_r = float(action), float(rew)
            steps += 1
            if reached:
                break

    # slice to actual length and return on-device tensors
    obs6 = states_buf[:steps].contiguous()
    actions = torch.tensor(actions_list, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
    beh_logits = torch.stack(beh_logits_buf, dim=0).to(device)

    return ExploreRollout(obs6=obs6, actions=actions, rewards=rewards, beh_logits=beh_logits, reuse_count=0)

def _first_demo_paths(ex_record_list: List[Dict], demo_root: str) -> List[str]:
    """Take all demos with phase==2, return their file paths (sorted by frames)."""
    demos_p2 = [r for r in ex_record_list if int(r.get("phase", 2)) == 2]
    demos_p2 = sorted(demos_p2, key=lambda r: r["frames"])
    return [os.path.join(demo_root, r["file"]) for r in demos_p2]

# --- load ONE phase-2 demo (no concatenation) ---
def _load_phase2_six_and_labels(demo_path: str, prev_action_start: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a single phase-2 expert demo and return:
      six:   (Te,6,H,W) float32 in [0,1] (boundary zeros; prev_action initialized from explore)
      y:     (Te,) int64 actions
    Boundary bit for explore->exploit will be set later when concatenating.
    """
    d = np.load(demo_path)
    obs = d["observations"].astype(np.float32) / 255.0
    acts = d["actions"].astype(np.int64)
    L = len(acts)
    boundary = np.zeros((L,), dtype=np.float32)
    six = build_six_from_demo_sequence(obs, acts, boundary, prev_action_start=prev_action_start)
    return torch.from_numpy(six).float(), torch.from_numpy(acts).long()

def _concat_explore_and_exploit(
    explore_six: torch.Tensor,       # (Tx,6,H,W) in [0,1]
    exploit_six: torch.Tensor,       # (Te,6,H,W) in [0,1]
    exploit_labels: torch.Tensor,    # (Te,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      obs6_cat:   (T,6,H,W)  explore followed by exploit
      labels_cat: (T,)       PAD on explore, expert labels on exploit
      exp_mask:   (T,)       1 on explore steps, 0 on exploit
      val_mask:   (T,)       1 everywhere (we're not padding in this path)
    Sets boundary bit=1 at the first exploitation frame.
    """
    Tx = explore_six.shape[0]
    Te = exploit_six.shape[0]
    # set boundary bit at first exploitation frame
    exploit_six = exploit_six.clone()
    if Te > 0:
        exploit_six[0, 5, :, :] = 1.0

    obs6_cat = torch.cat([explore_six, exploit_six], dim=0)        # (T,6,H,W)
    labels_cat = torch.cat([torch.full((Tx,), PAD_ACTION, dtype=torch.long, device=explore_six.device), exploit_labels], dim=0)
    exp_mask = torch.cat([torch.ones(Tx, device=explore_six.device), torch.zeros(Te, device=explore_six.device)], dim=0)
    val_mask = torch.ones(Tx + Te, device=explore_six.device)
    return obs6_cat, labels_cat, exp_mask, val_mask

def _kl_ess_should_refresh(
    cur_logits: torch.Tensor,    # (T,A)
    beh_logits: torch.Tensor,    # (T,A)
    kl_thr: float,
    ess_thr: float,
    is_clip_rho: float
) -> bool:
    kl = mean_kl_logits(cur_logits.detach(), beh_logits.detach()).item()
    if kl > kl_thr:
        return True
    # ESS check is done with taken actions when computing RL loss 
    return False

def run_training():
    args = parse_args()
    _setup_logging()
    start_time = datetime.datetime.utcnow()

    # Seeding
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.load_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        miss = getattr(ret, "missing_keys", [])
        unex = getattr(ret, "unexpected_keys", [])
        logger.info(f"[INIT] loaded BC init from {ck_path} (missing={len(miss)}, unexpected={len(unex)})")
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"[INIT] BC weights loaded from: {ck_path}")

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

    # Envs
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
    env.action_space.seed(args.seed)

    val_env = env  # reuse the same instance

    manifest = {
        "seed": args.seed,
        "start_time": start_time.isoformat() + "Z",
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "rl": {"gamma": args.gamma, "gae_lambda": args.gae_lambda, "use_gae": args.use_gae},
        "reuse": {
            "explore_reuse_M": args.explore_reuse_M,
            "offpolicy_correction": args.offpolicy_correction,
            "is_clip_rho": args.is_clip_rho,
            "kl_refresh_threshold": args.kl_refresh_threshold,
            "ess_refresh_ratio": args.ess_refresh_ratio,
        },
        "epoch_history": [],
    }

    patience = 0
    improved = False
    final_epoch = start_epoch

    # Per-epoch explore cache (discarded at epoch end)
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

        # compute how many batches this epoch: one pass over tasks
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
            # sample a batch of task indices
            start = b * args.batch_size
            end = min(len(train_tasks), (b + 1) * args.batch_size)
            batch_ids = [batch_indices[i] for i in range(start, end)]
            batch_tasks = [train_tasks[i] for i in batch_ids]

            for task in batch_tasks:
                tid = task["task_id"]
                cfg = MazeTaskManager.TaskConfig(**task["task_dict"])

                # Ensure we have a fresh or valid explore rollout
                if tid not in explore_cache or explore_cache[tid].reuse_count >= args.explore_reuse_M:
                    rollout = _collect_explore(policy_net, env, cfg, device, max_steps=250)
                    explore_cache[tid] = rollout
                else:
                    # KL-based refresh against current policy
                    with torch.no_grad():
                        seq = explore_cache[tid].obs6  # already on device
                        logits_all, values_all = policy_net(seq.unsqueeze(0))  # (1,Tx,A), (1,Tx,?)
                        cur_logits = logits_all[0] if logits_all.dim() == 3 else logits_all
                        if mean_kl_logits(cur_logits, explore_cache[tid].beh_logits).item() > args.kl_refresh_threshold:
                            rollout = _collect_explore(policy_net, env, cfg, device, max_steps=250)
                            explore_cache[tid] = rollout

                # -------- RL on explore chunk (computed ONCE) --------
                exp_six = explore_cache[tid].obs6.clone()            # (Tx,6,H,W) on device
                Tx = exp_six.shape[0]
                # forward on explore-only to get logits/values for RL
                logits_x_all, values_x_all = policy_net(exp_six.unsqueeze(0))
                cur_logits_x = logits_x_all[0] if logits_x_all.dim() == 3 else logits_x_all
                values_x = values_x_all
                if values_x.dim() == 3:
                    values_x = values_x[0, :, 0]
                elif values_x.dim() == 2:
                    values_x = values_x[0]
                else:
                    values_x = values_x.squeeze()
                actions_x = explore_cache[tid].actions
                rewards_x = explore_cache[tid].rewards
                beh_logits_x = explore_cache[tid].beh_logits

                # ESS refresh (based on taken actions)
                with torch.no_grad():
                    lp_cur = torch.log_softmax(cur_logits_x, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                    lp_beh = torch.log_softmax(beh_logits_x, dim=-1).gather(1, actions_x.unsqueeze(1)).squeeze(1)
                    rhos = torch.exp(lp_cur - lp_beh)
                    ess_ratio = ess_ratio_from_rhos(rhos).item()
                if args.explore_reuse_M > 1 and ess_ratio < args.ess_refresh_ratio:
                    # recollect explore and recompute
                    rollout = _collect_explore(policy_net, env, cfg, device, max_steps=250)
                    explore_cache[tid] = rollout
                    exp_six = explore_cache[tid].obs6.clone()
                    Tx = exp_six.shape[0]
                    logits_x_all, values_x_all = policy_net(exp_six.unsqueeze(0))
                    cur_logits_x = logits_x_all[0] if logits_x_all.dim() == 3 else logits_x_all
                    values_x = values_x_all
                    if values_x.dim() == 3:
                        values_x = values_x[0, :, 0]
                    elif values_x.dim() == 2:
                        values_x = values_x[0]
                    else:
                        values_x = values_x.squeeze()
                    actions_x = explore_cache[tid].actions
                    rewards_x = explore_cache[tid].rewards
                    beh_logits_x = explore_cache[tid].beh_logits

                loss_rl, ent_x, adv_abs = reinforce_with_baseline(
                    cur_logits=cur_logits_x,
                    actions=actions_x,
                    rewards=rewards_x,
                    values=values_x,
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                    use_gae=args.use_gae,
                    entropy_coef=args.explore_entropy_coef,
                    behavior_logits=beh_logits_x if args.offpolicy_correction != "none" else None,
                    offpolicy=args.offpolicy_correction,
                    is_clip_rho=args.is_clip_rho,
                )

                # -------- Batched BC over all demos (reuse SAME explore) --------
                if Tx > 0:
                    prev_action_start = float(explore_cache[tid].actions[-1].item())
                else:
                    prev_action_start = 0.0

                # Build sequences per demo first (explore + demo), then left-pad to same length and batch
                demo_obs_list: List[torch.Tensor] = []
                demo_lab_list: List[torch.Tensor] = []

                for demo_path in task["p2_paths"]:
                    p2_six, p2_labels = _load_phase2_six_and_labels(demo_path, prev_action_start=prev_action_start)
                    if p2_six.numel() == 0:
                        continue
                    # move demo to device before concat (exp_six already on device)
                    p2_six = p2_six.to(device, non_blocking=True)
                    p2_labels = p2_labels.to(device, non_blocking=True)

                    obs6_cat, labels_cat, _, _ = _concat_explore_and_exploit(exp_six, p2_six, p2_labels)
                    demo_obs_list.append(obs6_cat)     # (T_i,6,H,W) on device
                    demo_lab_list.append(labels_cat)   # (T_i,)      on device

                if len(demo_obs_list) == 0:
                    continue

                # left-pad to max length and stack -> (B_d, T_max, 6,H,W) and (B_d, T_max)
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
                    pad_obs.append(x)
                    pad_lab.append(y)

                batch_obs = torch.stack(pad_obs, dim=0)  # (B_d, T_max, 6,H,W) on device
                batch_lab = torch.stack(pad_lab, dim=0)  # (B_d, T_max)       on device

                optimizer.zero_grad()

                # single forward for all demos
                logits_b, _ = policy_net(batch_obs)      # (B_d, T_max, A)
                if args.label_smoothing > 0.0:
                    loss_bc = smoothed_cross_entropy(
                        logits_b, batch_lab,
                        ignore_index=PAD_ACTION, smoothing=args.label_smoothing
                    )
                else:
                    ce = nn.CrossEntropyLoss(ignore_index=PAD_ACTION)
                    loss_bc = ce(logits_b.reshape(-1, logits_b.size(-1)), batch_lab.reshape(-1))

                # Combine losses and step
                loss = args.rl_coef * loss_rl + loss_bc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                # bump reuse count and logs
                explore_cache[tid].reuse_count += 1
                running_train_loss += loss.item()
                running_bc += loss_bc.item()
                running_rl += loss_rl.item()
                running_ent += float(ent_x)
                count_updates += 1

            if count_updates > 0:
                pbar.set_postfix(loss=f"{running_train_loss/count_updates:.3f}",
                                 bc=f"{running_bc/count_updates:.3f}",
                                 rl=f"{running_rl/count_updates:.3f}",
                                 ent=f"{running_ent/count_updates:.3f}")

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
            if not args.disable_early_stop:
                patience += 1

        if not args.disable_early_stop and patience >= args.early_stop_patience and not improved_this:
            logger.info(f"[EARLY STOP] no improvement for {patience} epochs, stopping.")
            break

    if args.disable_early_stop and best_epoch != final_epoch:
        logger.info(f"[FINAL SAVE] saving final epoch {final_epoch}.")
        save_checkpoint(policy_net, final_epoch, best_val_score, args.save_path, args.load_path)

    # Save after EVERY epoch when early stopping is disabled
    if args.disable_early_stop and not improved_this:
        save_checkpoint(policy_net, epoch, best_val_score, args.save_path, args.load_path)

    logger.info("BC meta training complete.")

if __name__ == "__main__":
    run_training()
