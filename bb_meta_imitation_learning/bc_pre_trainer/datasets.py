# bc_pre_trainer/datasets.py

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    pad_or_truncate,
    PAD_ACTION,
    LEFT_ACTION, RIGHT_ACTION, STRAIGHT_ACTION,
    MIN_STRAIGHT_RECOVERY, MIN_COLLISION_RECOVERY, MIN_CORNER_STRAIGHT, MIN_STRAIGHT_SEGMENT,
    STEP_REWARD, COLLISION_REWARD,
)

def _augment_like_others(obs):
    """Apply brightness/noise like before, but return in the SAME dtype (preferably uint8)."""
    orig_dtype = obs.dtype
    obs_f = obs.astype(np.float32, copy=False)

    if random.random() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        obs_f = np.clip(obs_f * factor, 0, 255)

    if random.random() < 0.5:
        noise = np.random.randn(*obs_f.shape) * 5.0
        obs_f = np.clip(obs_f + noise, 0, 255)

    # Cast back to original dtype (saves memory if uint8)
    if np.issubdtype(orig_dtype, np.integer):
        return obs_f.astype(orig_dtype, copy=False)
    return obs_f  # already float32

class DemoDataset(Dataset):
    def __init__(self, demo_dir: str):
        self.files = sorted(glob.glob(os.path.join(demo_dir, '*.npz')))
        if not self.files:
            raise RuntimeError(f"No .npz demos found in {demo_dir}")
        print(f"[DemoDataset] {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        obs  = data['observations']                 # keep native dtype (likely uint8)
        acts = data['actions'].astype(np.int64)

        obs = _augment_like_others(obs)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)


class TurnSegmentDataset(Dataset):
    """Last forward → rotation burst → >= MIN_STRAIGHT_RECOVERY straights."""
    def __init__(self, demo_dir: str):
        self.segments = []
        for path in sorted(glob.glob(os.path.join(demo_dir, '*.npz'))):
            data = np.load(path)
            acts = data['actions']
            T = len(acts)
            i = 1
            while i < T - MIN_STRAIGHT_RECOVERY:
                if acts[i] in (LEFT_ACTION, RIGHT_ACTION):
                    j = i
                    while j < T and acts[j] in (LEFT_ACTION, RIGHT_ACTION):
                        j += 1
                    k, sc = j, 0
                    while k < T and sc < MIN_STRAIGHT_RECOVERY:
                        if acts[k] == STRAIGHT_ACTION:
                            sc += 1
                        else:
                            break
                        k += 1
                    if sc >= MIN_STRAIGHT_RECOVERY:
                        start, end = i - 1, k
                        self.segments.append((path, start, end))
                        i = k
                        continue
                i += 1

        if not self.segments:
            raise RuntimeError("No turn segments mined")
        print(f"[TurnSegmentDataset] {len(self.segments)} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        path, s, e = self.segments[idx]
        data  = np.load(path)
        obs   = data['observations'][s:e]          # keep native dtype
        acts  = data['actions'].astype(np.int64)[s:e]

        obs = _augment_like_others(obs)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)


class CollisionSegmentDataset(Dataset):
    """Collision reward → >= MIN_COLLISION_RECOVERY step rewards."""
    def __init__(self, demo_dir: str):
        self.segments = []
        for path in sorted(glob.glob(os.path.join(demo_dir, '*.npz'))):
            data = np.load(path)
            rews = data.get('rewards')
            if rews is None:
                continue
            T = len(rews)
            i = 0
            while i < T - MIN_COLLISION_RECOVERY:
                if (abs(rews[i] - COLLISION_REWARD) < 1e-4 and
                    abs(rews[i+1] - STEP_REWARD) < 1e-4):
                    j, rc = i+1, 0
                    while j < T and abs(rews[j] - STEP_REWARD) < 1e-4 and rc < MIN_COLLISION_RECOVERY:
                        rc += 1; j += 1
                    if rc >= MIN_COLLISION_RECOVERY:
                        self.segments.append((path, i, j))
                        i = j
                        continue
                i += 1

        if not self.segments:
            raise RuntimeError("No collision segments mined")
        print(f"[CollisionSegmentDataset] {len(self.segments)} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        path, s, e = self.segments[idx]
        data = np.load(path)
        obs  = data['observations'][s:e]           # keep native dtype
        acts = data['actions'].astype(np.int64)[s:e]

        obs = _augment_like_others(obs)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)


from numpy.lib.stride_tricks import sliding_window_view

class CornerSegmentDataset(Dataset):
    """Last rotation before >= MIN_CORNER_STRAIGHT straights."""
    def __init__(self, demo_dir: str, N_forward: int = MIN_CORNER_STRAIGHT):
        self.N = N_forward
        self.segments = []
        for path in sorted(glob.glob(os.path.join(demo_dir, '*.npz'))):
            data = np.load(path)
            acts = data['actions']
            T = len(acts)
            windows = sliding_window_view(acts, window_shape=self.N + 1)
            straight_mask = (windows[:, 1:] == STRAIGHT_ACTION).all(axis=1)

            idxs = np.nonzero(straight_mask)[0]
            for i in idxs:
                end = i + 1 + self.N
                j = i
                while j >= 0 and acts[j] in (LEFT_ACTION, RIGHT_ACTION):
                    j -= 1
                start = j + 1
                self.segments.append((path, start, end))

        if not self.segments:
            raise RuntimeError("No corner segments mined")
        print(f"[CornerSegmentDataset] {len(self.segments)} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        path, s, e = self.segments[idx]
        data = np.load(path)
        obs  = data['observations'][s:e]           # keep native dtype
        acts = data['actions'].astype(np.int64)[s:e]

        obs = _augment_like_others(obs)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)


class StraightSegmentDataset(Dataset):
    """
    Pure-straight clips: mines maximal runs of STRAIGHT with length >= MIN_STRAIGHT_SEGMENT.
    """
    def __init__(self, demo_dir: str, min_len: int = MIN_STRAIGHT_SEGMENT):
        self.min_len = min_len
        self.segments = []

        for path in sorted(glob.glob(os.path.join(demo_dir, '*.npz'))):
            data = np.load(path)
            acts = data['actions']
            T = len(acts)
            i = 0
            while i < T:
                if acts[i] == STRAIGHT_ACTION:
                    j = i
                    while j < T and acts[j] == STRAIGHT_ACTION:
                        j += 1
                    run_len = j - i
                    if run_len >= self.min_len:
                        self.segments.append((path, i, j))
                    i = j
                else:
                    i += 1

        if not self.segments:
            raise RuntimeError("No straight segments mined")
        print(f"[StraightSegmentDataset] {len(self.segments)} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        path, s, e = self.segments[idx]
        data = np.load(path)
        obs  = data['observations'][s:e]           # keep native dtype
        acts = data['actions'].astype(np.int64)[s:e]

        obs = _augment_like_others(obs)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)


def collate_fn(batch):
    obs_seq, act_seq = zip(*batch)
    # Convert to float32 & normalize HERE (saves memory in workers)
    np_img = np.stack(obs_seq, 0).astype(np.float32) / 255.0
    B, S, H, W, C = np_img.shape
    img   = torch.from_numpy(np_img).permute(0, 1, 4, 2, 3)
    extra = torch.zeros((B, S, 3, H, W), dtype=torch.float32)
    obs6  = torch.cat([img, extra], dim=2)
    acts  = torch.from_numpy(np.stack(act_seq, 0)).long()
    return obs6, acts
