# bc_pre_trainer/main.py

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from .train import train_bc

def main():
    p = argparse.ArgumentParser("BC Pre-Training w/ turn, collision & corner ups.")
    p.add_argument('--demo_root',          required=True)
    p.add_argument('--save_path',          required=True)
    p.add_argument('--action_dim', type=int, default=3)
    p.add_argument('--epochs',     type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--val_ratio',  type=float, default=0.1)
    p.add_argument('--early_stop_patience', type=int, default=5)
    p.add_argument('--seed',       type=int, default=42)
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
