# mri_train/data.py

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import random
import torch

from .utils import (
    PAD_ACTION,
    build_six_from_demo_sequence,
    apply_brightness_contrast,
    apply_gaussian_noise,
    apply_spatial_jitter,
)

def concat_explore_and_exploit(explore_six: torch.Tensor, exploit_six: torch.Tensor, exploit_labels: torch.Tensor):
    """
    Concatenate explore and exploit sequences along time on CPU.
    Explore part is padded with PAD_ACTION; exploit part gets the 'phase-2' flag in channel 5.
    """
    if explore_six.is_cuda or exploit_six.is_cuda or exploit_labels.is_cuda:
        raise RuntimeError("concat_explore_and_exploit expects CPU tensors; H2D is handled centrally.")
    Tx = explore_six.shape[0]
    Te = exploit_six.shape[0]
    exploit_six = exploit_six.clone()
    if Te > 0:
        exploit_six[:, 5, :, :] = 1.0
    obs6_cat = torch.cat([explore_six, exploit_six], dim=0)
    labels_cat = torch.cat([torch.full((Tx,), PAD_ACTION, dtype=torch.long), exploit_labels], dim=0)
    return obs6_cat, labels_cat

def first_demo_paths(ex_record_list: List[Dict], demo_root: str) -> List[str]:
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

def assert_start_goal_match(recs: List[Dict], tdict: Dict, tid: int):
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

def load_phase2_six_and_labels(
    demo_path: str,
    prev_action_start: float,
    preloaded: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """
    Load a single NPZ (observations [HWC uint8], actions int64), convert to SIX tensor + label tensor (CPU).
    If `preloaded` is provided, it should be (observations_uint8, actions_int64) as numpy arrays.
    """
    if preloaded is None:
        d = np.load(demo_path)
        obs = d["observations"]
        acts = d["actions"]
    else:
        obs, acts = preloaded

    obs = obs.astype(np.float32) / 255.0
    acts = acts.astype(np.int64)
    L = len(acts)
    boundary = np.zeros((L,), dtype=np.float32)
    six = build_six_from_demo_sequence(obs, acts, boundary, prev_action_start=prev_action_start)
    return torch.from_numpy(six).float(), torch.from_numpy(acts).long()

def maybe_augment_demo_six_cpu(p2_six_cpu: torch.Tensor, args) -> torch.Tensor:
    if not bool(getattr(args, "use_aug", False)):
        return p2_six_cpu
    if np.random.rand() > float(getattr(args, "aug_prob", 0.5)):
        return p2_six_cpu

    b_rng = getattr(args, "aug_brightness_range", (0.9, 1.1))
    c_rng = getattr(args, "aug_contrast_range", (0.9, 1.1))
    noise_std = float(getattr(args, "aug_noise_std", 0.02))
    jitter_max = int(getattr(args, "aug_jitter_max", 2))
    p_bc = float(getattr(args, "aug_bc_prob", 0.5))
    p_noise = float(getattr(args, "aug_noise_prob", 0.25))
    p_jitter = float(getattr(args, "aug_jitter_prob", 0.25))

    x = p2_six_cpu.numpy()
    imgs = x[:, 0:3, :, :]
    if np.random.rand() < p_bc:
        imgs = apply_brightness_contrast(imgs, brightness_range=b_rng, contrast_range=c_rng)
    if np.random.rand() < p_noise:
        imgs = apply_gaussian_noise(imgs, std=noise_std)
    if np.random.rand() < p_jitter:
        imgs = apply_spatial_jitter(imgs, max_shift=jitter_max)
    x[:, 0:3, :, :] = imgs
    return torch.from_numpy(x).float()

def select_demo_paths_for_task(recs_main, recs_syn, args, epoch: int) -> List[str]:
    paths = []
    if recs_main and getattr(args, "max_main_demos_per_task", 0) > 0:
        try:
            main_paths = first_demo_paths(recs_main, args.demo_root)
        except TypeError:
            main_paths = first_demo_paths(recs_main)
        random.shuffle(main_paths)
        paths.extend(main_paths[: int(getattr(args, "max_main_demos_per_task", 0))])

    use_syn = bool(getattr(args, "syn_demo_root", None)) and (epoch >= int(getattr(args, "syn_demo_min_epoch", 3)))
    if use_syn and recs_syn and getattr(args, "max_syn_demos_per_task", 0) > 0:
        try:
            syn_paths = first_demo_paths(recs_syn, args.syn_demo_root)
        except TypeError:
            syn_paths = first_demo_paths(recs_syn)
        random.shuffle(syn_paths)
        paths.extend(syn_paths[: int(getattr(args, "max_syn_demos_per_task", 0))])
    return paths
