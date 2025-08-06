# bc_pre_trainer/pre_trainer.py

import os
import sys
import glob
import random
import argparse
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torch import nn, optim

from snail_trpo.snail_model import SNAILPolicyValueNet

# ------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------
SEQ_LEN = 500
PAD_ACTION = -100

LEFT_ACTION     = 0
RIGHT_ACTION    = 1
STRAIGHT_ACTION = 2

MIN_STRAIGHT_RECOVERY = 3    # for turn & collision recovery
MIN_COLLISION_RECOVERY = 4
MIN_CORNER_STRAIGHT   = 4    # for corner segments

TURN_OVERSAMPLE_FRACTION      = 0.1
COLLISION_OVERSAMPLE_FRACTION = 0.1
CORNER_OVERSAMPLE_FRACTION    = 0.2

STEP_REWARD      = -0.01
COLLISION_PENALTY = -0.005
COLLISION_REWARD = STEP_REWARD + COLLISION_PENALTY  # -0.015

# ------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------
def pad_or_truncate(seq: np.ndarray, pad_value):
    L = seq.shape[0]
    if L >= SEQ_LEN:
        return seq[:SEQ_LEN]
    pad_len = SEQ_LEN - L
    pad_shape = (pad_len, *seq.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)

# ------------------------------------------------------------------------
# Full-demo dataset
# ------------------------------------------------------------------------
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
        obs  = data['observations'].astype(np.float32)
        acts = data['actions'].astype(np.int64)

        # augment
        if random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            obs = np.clip(obs * factor, 0, 255)
        if random.random() < 0.5:
            noise = np.random.randn(*obs.shape) * 5.0
            obs   = np.clip(obs + noise, 0, 255)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)

# ------------------------------------------------------------------------
# Turn + straight-recovery segments, starting at the last forward before a rotation burst
# ------------------------------------------------------------------------
class TurnSegmentDataset(Dataset):
    def __init__(self, demo_dir: str):
        self.segments = []
        for path in sorted(glob.glob(os.path.join(demo_dir, '*.npz'))):
            data = np.load(path)
            acts = data['actions']
            T = len(acts)
            i = 1  # start at 1 so we can back up one
            while i < T - MIN_STRAIGHT_RECOVERY:
                # detect cluster: at least one rotation at i, followed by rotations, then straights
                if acts[i] in (LEFT_ACTION, RIGHT_ACTION):
                    # find end of rotation cluster
                    j = i
                    while j < T and acts[j] in (LEFT_ACTION, RIGHT_ACTION):
                        j += 1
                    # check for straight recovery after
                    k = j
                    sc = 0
                    while k < T and sc < MIN_STRAIGHT_RECOVERY:
                        if acts[k] == STRAIGHT_ACTION:
                            sc += 1
                        else:
                            break
                        k += 1
                    if sc >= MIN_STRAIGHT_RECOVERY:
                        # start one step before i (last forward)
                        start = i - 1
                        end   = k
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
        obs   = data['observations'].astype(np.float32)[s:e]
        acts  = data['actions'].astype(np.int64)[s:e]

        # same augmentations
        if random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            obs = np.clip(obs * factor, 0, 255)
        if random.random() < 0.5:
            noise = np.random.randn(*obs.shape) * 5.0
            obs   = np.clip(obs + noise, 0, 255)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)

# ------------------------------------------------------------------------
# Collision + recovery segments
# ------------------------------------------------------------------------
class CollisionSegmentDataset(Dataset):
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
                if abs(rews[i] - COLLISION_REWARD) < 1e-4 and abs(rews[i+1] - STEP_REWARD) < 1e-4:
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
        obs  = data['observations'].astype(np.float32)[s:e]
        acts = data['actions'].astype(np.int64)[s:e]

        if random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            obs = np.clip(obs * factor, 0, 255)
        if random.random() < 0.5:
            noise = np.random.randn(*obs.shape) * 5.0
            obs   = np.clip(obs + noise, 0, 255)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)

# ------------------------------------------------------------------------
# Corner-turn + straight segments
# ------------------------------------------------------------------------
class CornerSegmentDataset(Dataset):
    def __init__(self, demo_dir: str, N_forward: int = MIN_CORNER_STRAIGHT):
        self.N = N_forward
        self.segments = []
        for path in sorted(glob.glob(os.path.join(demo_dir, '*.npz'))):
            data = np.load(path)
            acts = data['actions']
            T = len(acts)
            i = 0
            while i < T - (1 + self.N):
                # require N straight steps immediately
                if np.all(acts[i+1:i+1+self.N] == STRAIGHT_ACTION):
                    # back up to last rotation
                    j = i
                    while j >= 0 and acts[j] in (LEFT_ACTION, RIGHT_ACTION):
                        j -= 1
                    start = j + 1
                    end   = start + 1 + self.N
                    if end <= T:
                        self.segments.append((path, start, end))
                        i = end
                        continue
                i += 1

        if not self.segments:
            raise RuntimeError("No corner segments mined")
        print(f"[CornerSegmentDataset] {len(self.segments)} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        path, s, e = self.segments[idx]
        data = np.load(path)
        obs  = data['observations'].astype(np.float32)[s:e]
        acts = data['actions'].astype(np.int64)[s:e]

        if random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            obs = np.clip(obs * factor, 0, 255)
        if random.random() < 0.5:
            noise = np.random.randn(*obs.shape) * 5.0
            obs   = np.clip(obs + noise, 0, 255)

        return pad_or_truncate(obs, 0), pad_or_truncate(acts, PAD_ACTION)

# ------------------------------------------------------------------------
# Collate fn
# ------------------------------------------------------------------------
def collate_fn(batch):
    obs_seq, act_seq = zip(*batch)
    np_img = np.stack(obs_seq, 0).astype(np.float32) / 255.0
    B, S, H, W, C = np_img.shape
    img   = torch.from_numpy(np_img).permute(0,1,4,2,3)
    extra = torch.zeros((B, S, 3, H, W), dtype=torch.float32)
    obs6  = torch.cat([img, extra], dim=2)
    acts  = torch.from_numpy(np.stack(act_seq, 0)).long()
    return obs6, acts

# ------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------
def train_bc(
    demo_root, save_path, action_dim, epochs, batch_size,
    lr, val_ratio, patience_max, seed
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark    = False

    # load full demos
    mf = os.path.join(demo_root, 'demo_manifest.csv')
    if not os.path.isfile(mf):
        raise FileNotFoundError(f"Missing manifest: {mf}")
    demo_dir   = os.path.join(demo_root, 'demos')
    full_ds    = DemoDataset(demo_dir)
    N_full     = len(full_ds)
    n_val      = int(val_ratio * N_full)
    n_train    = N_full - n_val
    train_full, val_full = random_split(full_ds, [n_train, n_val])

    # mine segments
    turn_ds   = TurnSegmentDataset(demo_dir)
    coll_ds   = CollisionSegmentDataset(demo_dir)
    corner_ds = CornerSegmentDataset(demo_dir)

    n_full   = len(train_full)
    n_turn   = len(turn_ds)
    n_coll   = len(coll_ds)
    n_corner = len(corner_ds)

    # compute weights
    T = TURN_OVERSAMPLE_FRACTION
    C = COLLISION_OVERSAMPLE_FRACTION
    X = CORNER_OVERSAMPLE_FRACTION
    F = 1.0 - (T + C + X)
    if F <= 0:
        raise ValueError("Oversample fractions sum >= 1.0")
    w_full   = 1.0
    w_turn   = (T / F) * (n_full / n_turn)
    w_coll   = (C / F) * (n_full / n_coll)
    w_corner = (X / F) * (n_full / n_corner)

    weights = ([w_full]*n_full +
               [w_turn]*n_turn +
               [w_coll]*n_coll +
               [w_corner]*n_corner)
    combined = ConcatDataset([train_full, turn_ds, coll_ds, corner_ds])
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(combined, batch_size=batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_full, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = SNAILPolicyValueNet(action_dim=action_dim,
                                  base_dim=256,
                                  policy_filters=32,
                                  policy_attn_dim=16,
                                  value_filters=16,
                                  seq_len=SEQ_LEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD_ACTION)

    best_val = float('inf')
    patience  = 0
    for epoch in range(1, epochs+1):
        model.train()
        acc_loss = 0.0
        for obs6, acts in train_loader:
            obs6, acts = obs6.to(device), acts.to(device)
            optimizer.zero_grad()
            logits, _ = model(obs6)
            loss = loss_fn(logits.view(-1, action_dim), acts.view(-1))
            loss.backward()
            optimizer.step()
            acc_loss += loss.item() * acts.numel()
        train_ce = acc_loss / (len(weights) * SEQ_LEN)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs6, acts in val_loader:
                obs6, acts = obs6.to(device), acts.to(device)
                logits, _ = model(obs6)
                l = loss_fn(logits.view(-1, action_dim), acts.view(-1))
                val_loss += l.item() * acts.numel()
        val_ce = val_loss / (n_val * SEQ_LEN)

        print(f"[Epoch {epoch:02d}] train_ce={train_ce:.4f}  val_ce={val_ce:.4f}")

        if val_ce < best_val:
            best_val = val_ce
            patience  = 0
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'bc_best.pth'))
        else:
            patience += 1
            if patience >= patience_max:
                print(f"Early stopping @ epoch {epoch}")
                break

    os.makedirs(save_path, exist_ok=True)
    manifest = {
        'total_full':       N_full,
        'train_full':       n_full,
        'val_full':         n_val,
        'turn_segments':    n_turn,
        'collision_segments': n_coll,
        'corner_segments':  n_corner,
        'seq_len':          SEQ_LEN,
        'turn_frac':        TURN_OVERSAMPLE_FRACTION,
        'coll_frac':        COLLISION_OVERSAMPLE_FRACTION,
        'corner_frac':      CORNER_OVERSAMPLE_FRACTION
    }
    with open(os.path.join(save_path, 'pretrain_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] best val_ce={best_val:.4f}")
    print(f"→ checkpoint: {save_path}/bc_best.pth")
    print(f"→ manifest:   {save_path}/pretrain_manifest.json")

def main():
    p = argparse.ArgumentParser("BC Pre-Training w/ turn, collision & corner ups.")
    p.add_argument('--demo_root', required=True)
    p.add_argument('--save_path', required=True)
    p.add_argument('--action_dim', type=int, default=3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--early_stop_patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    train_bc(
        args.demo_root,
        args.save_path,
        args.action_dim,
        args.epochs,
        args.batch_size,
        args.lr,
        args.val_ratio,
        args.early_stop_patience,
        args.seed
    )

if __name__ == '__main__':
    main()
