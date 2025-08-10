# bc_pre_trainer/train.py

import os
import torch
import json
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torch import optim, nn
from tqdm.auto import tqdm

from .utils import SEQ_LEN, PAD_ACTION
from .datasets import (
    DemoDataset, TurnSegmentDataset,
    CollisionSegmentDataset, CornerSegmentDataset, StraightSegmentDataset,
    collate_fn
)
from bb_meta_imitation_learning.snail_trpo.snail_model import SNAILPolicyValueNet


def train_bc(
    demo_root, save_path, action_dim, epochs, batch_size,
    lr, val_ratio, patience_max, seed
):
    # reproducibility
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark    = False

    # load full demos
    mf = os.path.join(demo_root, 'demo_manifest.csv')
    if not os.path.isfile(mf):
        raise FileNotFoundError(f"Missing manifest: {mf}")
    demo_dir = os.path.join(demo_root, 'demos')
    full_ds  = DemoDataset(demo_dir)
    N_full   = len(full_ds)
    n_val    = int(val_ratio * N_full)
    n_train  = N_full - n_val
    train_full, val_full = random_split(full_ds, [n_train, n_val])

    # mine segments
    turn_ds     = TurnSegmentDataset(demo_dir)
    coll_ds     = CollisionSegmentDataset(demo_dir)
    corner_ds   = CornerSegmentDataset(demo_dir)
    straight_ds = StraightSegmentDataset(demo_dir)

    n_full     = len(train_full)
    n_turn     = len(turn_ds)
    n_coll     = len(coll_ds)
    n_corner   = len(corner_ds)
    n_straight = len(straight_ds)

    # compute mixing weights
    from .utils import (
        TURN_OVERSAMPLE_FRACTION,
        COLLISION_OVERSAMPLE_FRACTION,
        CORNER_OVERSAMPLE_FRACTION,
        STRAIGHT_OVERSAMPLE_FRACTION
    )
    T = TURN_OVERSAMPLE_FRACTION
    C = COLLISION_OVERSAMPLE_FRACTION
    X = CORNER_OVERSAMPLE_FRACTION
    S = STRAIGHT_OVERSAMPLE_FRACTION
    F = 1.0 - (T + C + X + S)
    if F <= 0:
        raise ValueError("Oversample fractions sum >= 1.0")

    w_full     = 1.0
    w_turn     = (T / F) * (n_full / n_turn)
    w_coll     = (C / F) * (n_full / n_coll)
    w_corner   = (X / F) * (n_full / n_corner)
    w_straight = (S / F) * (n_full / n_straight)

    weights = (
        [w_full]     * n_full     +
        [w_turn]     * n_turn     +
        [w_coll]     * n_coll     +
        [w_corner]   * n_corner   +
        [w_straight] * n_straight
    )
    combined = ConcatDataset([train_full, turn_ds, coll_ds, corner_ds, straight_ds])
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # —— DataLoaders (memory-safer) ——
    on_cuda = torch.cuda.is_available()
    pin     = on_cuda  # only pin when CUDA is present

    # Smaller val batch helps peak RAM specifically where you saw the crash
    val_bs = min(batch_size, 8)

    train_loader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,                  # keep modest
        prefetch_factor=2,              # only used when num_workers>0
        persistent_workers=True,        # keep workers alive between epochs
        pin_memory=pin
    )
    val_loader = DataLoader(
        val_full,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,                  # keep validation lean on RAM
        pin_memory=pin
    )

    # model, optimizer, loss
    device     = torch.device('cuda' if on_cuda else 'cpu')
    policy_net = SNAILPolicyValueNet(
        action_dim=action_dim,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=SEQ_LEN
    ).to(device)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD_ACTION)

    # resume support
    os.makedirs(save_path, exist_ok=True)
    resume_path  = os.path.join(save_path, "pretrain_ckp_load.pth")
    start_epoch  = 1
    best_val     = float('inf')
    patience     = 0

    if os.path.isfile(resume_path):
        print(f"[Resuming] loading checkpoint {resume_path}")
        ck = torch.load(resume_path, map_location=device)
        policy_net.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['opt_state'])
        start_epoch = ck['epoch'] + 1
        best_val     = ck.get('best_val', best_val)
        patience     = ck.get('patience', patience)
        print(f" → Resumed at epoch {ck['epoch']}, best_val={best_val:.4f}, patience={patience}")

    # training loop w/ tqdm
    epoch_bar = tqdm(range(start_epoch, epochs+1), desc="Epoch", unit="ep")
    for epoch in epoch_bar:
        # training phase
        policy_net.train()
        acc_loss  = 0.0
        batch_bar = tqdm(train_loader, desc=f"Train Ep{epoch}", leave=False, unit="batch")
        for obs6, acts in batch_bar:
            obs6, acts = obs6.to(device), acts.to(device)
            optimizer.zero_grad()
            logits, _ = policy_net(obs6)
            loss = loss_fn(logits.view(-1, action_dim), acts.view(-1))
            loss.backward()
            optimizer.step()
            acc_loss += loss.item() * acts.numel()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_ce = acc_loss / (len(weights) * SEQ_LEN)

        # validation phase (full demos only)
        policy_net.eval()
        val_loss = 0.0
        for obs6, acts in tqdm(val_loader, desc=f" Val Ep{epoch}", leave=False, unit="batch"):
            obs6, acts = obs6.to(device), acts.to(device)
            with torch.no_grad():
                logits, _ = policy_net(obs6)
                l = loss_fn(logits.view(-1, action_dim), acts.view(-1))
            val_loss += l.item() * acts.numel()
        val_ce = val_loss / (n_val * SEQ_LEN)

        # update epoch bar with CE metrics
        epoch_bar.set_postfix(train_ce=f"{train_ce:.4f}", val_ce=f"{val_ce:.4f}")

        # checkpoint this epoch & update resume pointer
        ckpt = {
            'epoch':       epoch,
            'model_state': policy_net.state_dict(),
            'opt_state':   optimizer.state_dict(),
            'best_val':    best_val,
            'patience':    patience,
        }
        torch.save(ckpt, os.path.join(save_path, f"ckpt_epoch_{epoch}.pth"))
        torch.save(ckpt, resume_path)

        # best‐model saving & early‐stop logic
        if val_ce < best_val:
            best_val = val_ce
            patience  = 0
            torch.save(policy_net.state_dict(), os.path.join(save_path, 'bc_best.pth'))
        else:
            patience += 1
            if patience >= patience_max:
                print(f"Early stopping @ epoch {epoch}")
                break

    # write manifest
    manifest = {
        'total_full':         N_full,
        'train_full':         n_full,
        'val_full':           n_val,
        'turn_segments':      n_turn,
        'collision_segments': n_coll,
        'corner_segments':    n_corner,
        'straight_segments':  n_straight,
        'seq_len':            SEQ_LEN,
        'turn_frac':          T,
        'coll_frac':          C,
        'corner_frac':        X,
        'straight_frac':      S,
    }
    with open(os.path.join(save_path, 'pretrain_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] best val_ce={best_val:.4f}")
    print(f"→ bc_best.pth      in {save_path}")
    print(f"→ last ckpt for resume: pretrain_ckp_load.pth")
