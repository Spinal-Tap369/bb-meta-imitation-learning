# bc_pre_trainer/train.py

import os
import torch
import json
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torch import optim, nn

from bc_pre_trainer.utils import SEQ_LEN, PAD_ACTION
from bc_pre_trainer.datasets import (
    DemoDataset, TurnSegmentDataset,
    CollisionSegmentDataset, CornerSegmentDataset, collate_fn
)
from snail_trpo.snail_model import SNAILPolicyValueNet

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
    turn_ds   = TurnSegmentDataset(demo_dir)
    coll_ds   = CollisionSegmentDataset(demo_dir)
    corner_ds = CornerSegmentDataset(demo_dir)

    n_full   = len(train_full)
    n_turn   = len(turn_ds)
    n_coll   = len(coll_ds)
    n_corner = len(corner_ds)

    # compute mixing weights
    from .utils import (
        TURN_OVERSAMPLE_FRACTION,
        COLLISION_OVERSAMPLE_FRACTION,
        CORNER_OVERSAMPLE_FRACTION
    )
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

    # model, optimizer, loss
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    best_val = float('inf')
    patience  = 0
    for epoch in range(1, epochs+1):
        policy_net.train()
        acc_loss = 0.0
        for obs6, acts in train_loader:
            obs6, acts = obs6.to(device), acts.to(device)
            optimizer.zero_grad()
            logits, _ = policy_net(obs6)
            loss = loss_fn(logits.view(-1, action_dim), acts.view(-1))
            loss.backward()
            optimizer.step()
            acc_loss += loss.item() * acts.numel()
        train_ce = acc_loss / (len(weights) * SEQ_LEN)

        # validation
        policy_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs6, acts in val_loader:
                obs6, acts = obs6.to(device), acts.to(device)
                logits, _ = policy_net(obs6)
                l = loss_fn(logits.view(-1, action_dim), acts.view(-1))
                val_loss += l.item() * acts.numel()
        val_ce = val_loss / (n_val * SEQ_LEN)

        print(f"[Epoch {epoch:02d}] train_ce={train_ce:.4f}  val_ce={val_ce:.4f}")

        if val_ce < best_val:
            best_val = val_ce
            patience  = 0
            os.makedirs(save_path, exist_ok=True)
            torch.save(policy_net.state_dict(), os.path.join(save_path, 'bc_best.pth'))
        else:
            patience += 1
            if patience >= patience_max:
                print(f"Early stopping @ epoch {epoch}")
                break

    # write manifest
    os.makedirs(save_path, exist_ok=True)
    manifest = {
        'total_full':       N_full,
        'train_full':       n_full,
        'val_full':         n_val,
        'turn_segments':    n_turn,
        'collision_segments': n_coll,
        'corner_segments':  n_corner,
        'seq_len':          SEQ_LEN,
        'turn_frac':        T,
        'coll_frac':        C,
        'corner_frac':      X
    }
    with open(os.path.join(save_path, 'pretrain_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] best val_ce={best_val:.4f}")
    print(f"→ checkpoint: {save_path}/bc_best.pth")
    print(f"→ manifest:   {save_path}/pretrain_manifest.json")
