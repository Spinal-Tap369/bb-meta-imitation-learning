# bc_meta_train_vec/train_utils.py

import os
import json
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym

from bb_meta_imitation_learning.env.maze_task import MazeTaskManager

# Constants
SEQ_LEN = 500
PAD_ACTION = -100
MAX_VALIDATION_STEPS = 1000


def apply_brightness_contrast(
    img_seq: np.ndarray,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
):
    """Random brightness and contrast adjustment on a sequence (T,C,H,W)."""
    b_factor = np.random.uniform(*brightness_range)
    c_factor = np.random.uniform(*contrast_range)
    seq = img_seq * b_factor
    mean = seq.mean(axis=(2, 3), keepdims=True)
    seq = (seq - mean) * c_factor + mean
    return np.clip(seq, 0.0, 1.0)


def apply_gaussian_noise(img_seq: np.ndarray, std=0.02):
    """Add Gaussian noise to a sequence (T,C,H,W)."""
    noise = np.random.normal(scale=std, size=img_seq.shape).astype(np.float32)
    return np.clip(img_seq + noise, 0.0, 1.0)


def apply_spatial_jitter(img_seq: np.ndarray, max_shift=2):
    """Apply integer pixel shifts within ±max_shift on H and W."""
    T, C, H, W = img_seq.shape
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    jittered = np.zeros_like(img_seq)
    src_y1, src_y2 = max(0, -shift_y), min(H, H - shift_y)
    src_x1, src_x2 = max(0, -shift_x), min(W, W - shift_x)
    dst_y1, dst_y2 = max(0, shift_y), min(H, H + shift_y)
    dst_x1, dst_x2 = max(0, shift_x), min(W, W + shift_x)
    jittered[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = img_seq[:, :, src_y1:src_y2, src_x1:src_x2]
    return jittered


def temporal_subsequence_and_pad(
    obs_six: np.ndarray,
    actions: np.ndarray,
    explore_mask: np.ndarray,
    valid_mask: np.ndarray,
    seq_len=SEQ_LEN,
    min_window=200,
    attempt_limit=10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random temporal crop with left-padding; preserves masks and boundary bit."""
    L_full = obs_six.shape[0]
    assert L_full == seq_len, "Expected full-length input for cropping logic"

    if (actions != PAD_ACTION).sum() == 0:
        return obs_six, actions, explore_mask, valid_mask

    for _ in range(attempt_limit):
        L = random.randint(min_window, seq_len)
        start = random.randint(0, seq_len - L)
        window_actions = actions[start : start + L]
        if (window_actions != PAD_ACTION).any():
            obs_sub = obs_six[start : start + L]
            act_sub = actions[start : start + L]
            exp_sub = explore_mask[start : start + L]
            val_sub = valid_mask[start : start + L]

            pad_len = seq_len - L
            C, H, W = obs_six.shape[1:]
            pad_obs = np.zeros((pad_len, C, H, W), dtype=obs_six.dtype)
            pad_actions = np.full((pad_len,), PAD_ACTION, dtype=actions.dtype)
            pad_exp = np.zeros((pad_len,), dtype=exp_sub.dtype)
            pad_val = np.zeros((pad_len,), dtype=val_sub.dtype)

            new_obs = np.concatenate([pad_obs, obs_sub], axis=0)
            new_actions = np.concatenate([pad_actions, act_sub], axis=0)
            new_exp_mask = np.concatenate([pad_exp, exp_sub], axis=0)
            new_val_mask = np.concatenate([pad_val, val_sub], axis=0)

            exploit_mask = new_actions != PAD_ACTION
            if exploit_mask.any():
                first_idx = int(np.argmax(exploit_mask))
                if np.all(new_obs[first_idx, 5] == 0.0):
                    new_obs[first_idx, 5, :, :] = 1.0

            return new_obs, new_actions, new_exp_mask, new_val_mask

    return obs_six, actions, explore_mask, valid_mask


def smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = PAD_ACTION,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Label-smoothed CE that ignores targets == ignore_index."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    mask = targets != ignore_index
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    targets_safe = torch.where(mask, targets, torch.zeros_like(targets))
    true_lp = torch.gather(log_probs, -1, targets_safe.unsqueeze(-1)).squeeze(-1)
    uni_lp = log_probs.mean(dim=-1)
    loss_tok = -(1.0 - smoothing) * true_lp - smoothing * uni_lp

    loss = (loss_tok * mask).sum() / mask.sum()
    return loss


def make_task_id(task_dict: Dict) -> int:
    """Compute a stable task identifier from task fields."""
    if "task_idx" in task_dict:
        return int(task_dict["task_idx"])
    key = (
        tuple(task_dict.get("start", [])),
        tuple(task_dict.get("goal", [])),
        json.dumps(task_dict.get("cell_walls", []), sort_keys=True),
        json.dumps(task_dict.get("cell_texts", []), sort_keys=True),
        task_dict.get("cell_size", -1),
        task_dict.get("wall_height", -1),
    )
    return abs(hash(key)) % (1 << 31)


def build_six_from_demo_sequence(
    obs_rgb_seq: np.ndarray,
    action_seq: np.ndarray,
    boundary_mask: np.ndarray,
    prev_action_start: float = 0.0,
) -> np.ndarray:
    """Build (T,6,H,W) from RGB (T,H,W,3) with prev-action, prev-reward, and boundary channels."""
    T, H, W, _ = obs_rgb_seq.shape
    six = []
    prev_action, prev_reward = prev_action_start, 0.0
    for t in range(T):
        img = obs_rgb_seq[t].transpose(2, 0, 1)
        pa = np.full((1, H, W), prev_action, dtype=np.float32)
        pr = np.full((1, H, W), prev_reward, dtype=np.float32)
        bb = np.full((1, H, W), float(boundary_mask[t]), dtype=np.float32)
        six.append(np.concatenate([img, pa, pr, bb], axis=0))
        prev_action = float(action_seq[t])
    return np.stack(six)


def assemble_meta_sequence(
    explore_files: List[str],
    exploit_file: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble exploration + exploitation into fixed-length tensors."""
    parts_six, parts_act, parts_exp, parts_valid, parts_boundary = [], [], [], [], []
    prev_action = 0.0

    # Exploration (unlabeled)
    total_explore_len = 0
    for ef in explore_files:
        d = np.load(ef)
        obs, acts = d["observations"].astype(np.float32), d["actions"].astype(np.int64)
        L = len(acts)
        boundary = np.zeros((L,), dtype=np.float32)
        six = build_six_from_demo_sequence(obs, acts, boundary, prev_action)

        parts_six.append(six)
        parts_act.append(np.full((L,), PAD_ACTION, dtype=np.int64))
        parts_exp.append(np.ones((L,), dtype=np.float32))
        parts_valid.append(np.ones((L,), dtype=np.float32))
        parts_boundary.append(boundary)

        prev_action = float(acts[-1]) if L > 0 else prev_action
        total_explore_len += L

    # Exploitation (labeled)
    de = np.load(exploit_file)
    obs_e, acts_e = de["observations"].astype(np.float32), de["actions"].astype(np.int64)
    L_e = len(acts_e)
    boundary_e = np.zeros((L_e,), dtype=np.float32)
    if L_e > 0:
        boundary_e[0] = 1.0

    six_e = build_six_from_demo_sequence(obs_e, acts_e, boundary_e, prev_action)

    parts_six.append(six_e)
    parts_act.append(acts_e)
    parts_exp.append(np.zeros((L_e,), dtype=np.float32))
    parts_valid.append(np.ones((L_e,), dtype=np.float32))
    parts_boundary.append(boundary_e)

    # Concatenate and pad/trim
    combined_six = np.concatenate(parts_six, axis=0)
    combined_act = np.concatenate(parts_act, axis=0)
    combined_exp = np.concatenate(parts_exp, axis=0)
    combined_valid = np.concatenate(parts_valid, axis=0)

    L = len(combined_act)
    if L > seq_len:
        start = L - seq_len
        obs_six = combined_six[start:]
        actions = combined_act[start:]
        exp_mask = combined_exp[start:]
        val_mask = combined_valid[start:]
        exploit_mask = actions != PAD_ACTION
        if exploit_mask.any():
            idx = int(np.argmax(exploit_mask))
            if np.all(obs_six[idx, 5] == 0.0):
                obs_six[idx, 5, :, :] = 1.0
    else:
        pad_len = seq_len - L
        C, H, W = combined_six.shape[1:]
        obs_six = np.concatenate([combined_six, np.zeros((pad_len, C, H, W), dtype=np.float32)], axis=0)
        actions = np.concatenate([combined_act, np.full((pad_len,), PAD_ACTION, dtype=np.int64)], axis=0)
        exp_mask = np.concatenate([combined_exp, np.zeros((pad_len,), dtype=np.float32)], axis=0)
        val_mask = np.concatenate([combined_valid, np.zeros((pad_len,), dtype=np.float32)], axis=0)

    return obs_six, actions, exp_mask, val_mask


def load_all_manifests(demo_root: str) -> Dict[int, List[Dict]]:
    """Load demo and DAgger manifests into {task_idx: [records]}."""
    demos: Dict[int, List[Dict]] = {}

    def _load(path):
        if not os.path.isfile(path):
            return
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                tid = int(row["task_idx"])
                rec = {
                    "demo_id": int(row["demo_id"]),
                    "task_idx": tid,
                    "phase": int(row["phase"]),
                    "frames": int(row["frames"]),
                    "file": row["file"],
                    "start_x": row.get("start_x"),
                    "start_y": row.get("start_y"),
                    "goal_x": row.get("goal_x"),
                    "goal_y": row.get("goal_y"),
                }
                demos.setdefault(tid, []).append(rec)

    _load(os.path.join(demo_root, "demo_manifest.csv"))
    _load(os.path.join(demo_root, "dagger_manifest.csv"))
    return demos


def eval_sampled_val(policy_net, val_task_entries, env, device, sample_n=3, max_steps=MAX_VALIDATION_STEPS):
    """Roll out on sampled validation tasks and aggregate metrics.

    Boundary channel (index 5) is 1.0 for all timesteps that occur in phase 2,
    and 0.0 in phase 1. Start/goal are NOT randomized — they come from the task config.
    """
    policy_net.eval()
    sampled = random.sample(val_task_entries, min(sample_n, len(val_task_entries)))
    results = []

    with torch.no_grad():
        pbar = tqdm(sampled, desc="validation", leave=False)
        for ex in pbar:
            tid = ex["task_id"]
            cfg = MazeTaskManager.TaskConfig(**ex["task_dict"])
            env.unwrapped.set_task(cfg)  # fixed start/goal from train_trials.json

            obs, _ = env.reset()
            done, trunc = False, False
            last_a, last_r = 0.0, 0.0
            steps = 0
            reached = False

            states = []  # (t, 6, H, W)

            while not done and not trunc and steps < max_steps:
                cur_p = env.unwrapped.maze_core.phase
                bb = 1.0 if cur_p == 2 else 0.0  # persistent phase-2 boundary bit

                img = obs.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)
                H, W = img.shape[1], img.shape[2]
                c3 = np.full((1, H, W), last_a, dtype=np.float32)  # prev action
                c4 = np.full((1, H, W), last_r, dtype=np.float32)  # prev reward
                c5 = np.full((1, H, W), bb,     dtype=np.float32)  # boundary bit
                obs6 = np.concatenate([img, c3, c4, c5], axis=0)
                states.append(obs6)

                seq = torch.from_numpy(np.stack(states, axis=0)[None]).float().to(device)  # (1, t, 6, H, W)
                logits, _ = policy_net.act_single_step(seq)
                action = torch.distributions.Categorical(logits=logits).sample().item()

                obs, rew, done, trunc, info = env.step(action)
                if env.unwrapped.maze_core.phase_metrics[2]["goal_rewards"] > 0:
                    reached = True

                last_a, last_r = float(action), float(rew)
                steps += 1
                if reached:
                    break

            p1 = env.unwrapped.maze_core.phase_metrics[1]["steps"]
            p2 = env.unwrapped.maze_core.phase_metrics[2]["steps"]
            total = p1 + p2
            succ = int(reached)

            trans = False
            if p1 >= env.unwrapped.maze_core.phase_step_limit and p2 == 0 and not succ:
                trans = True

            results.append(
                {
                    "task_id": tid,
                    "phase1": p1,
                    "phase2": p2,
                    "success": succ,
                    "total_steps": total,
                    "transition_issue": trans,
                }
            )
            pbar.set_postfix(task=tid, p1=p1, p2=p2, succ=succ)

    phase1s = [r["phase1"] for r in results]
    phase2s = [r["phase2"] for r in results]
    totals  = [r["total_steps"] for r in results]
    succs   = [r["success"] for r in results]

    return (
        results,
        float(np.mean(phase1s)),
        float(np.mean(phase2s)),
        float(np.std(phase1s)),
        float(np.std(phase2s)),
        float(np.mean(totals)),
        float(np.mean(succs)),
    )
