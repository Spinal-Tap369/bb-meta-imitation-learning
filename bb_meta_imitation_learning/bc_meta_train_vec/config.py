# bc_meta_train_vec/config.py

import argparse

def parse_args():
    p = argparse.ArgumentParser("BC Meta-Imitation Training (SNAIL) — MRI-style (vectorized)")

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo_root", type=str, required=True, help="Root with demo_manifest.csv, dagger_manifest.csv, and demos/")
    p.add_argument("--train_trials", type=str, required=True, help="Path to train_trials.json")
    p.add_argument("--bc_init", type=str, default=None, help="Optional BC pretrain checkpoint (state_dict)")
    p.add_argument("--save_path", type=str, default="bc_meta_ckpts")
    p.add_argument("--load_path", type=str, default="bc_meta_ckpts_load")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--nbc", type=int, default=8, help="BC refinement steps per collected explore rollout")

    # Loss / regularization
    p.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for exploitation BC")
    p.add_argument("--explore_entropy_coef", type=float, default=5e-3, help="Entropy coefficient on exploration")
    p.add_argument("--rl_coef", type=float, default=1.0, help="Weight on the RL loss (exploration)")

    # RL (REINFORCE + baseline)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=1.0, help="GAE lambda (1.0 = Monte Carlo)")
    p.add_argument("--use_gae", action="store_true", help="Use GAE(lambda) on exploration")

    # Exploration rollout reuse
    p.add_argument("--explore_reuse_M", type=int, default=1, help="Reuse count for the same phase-1 rollout (1=no reuse)")
    p.add_argument("--offpolicy_correction", type=str, default="none", choices=["none", "is", "vtrace"], help="Off-policy correction when reusing")
    p.add_argument("--is_clip_rho", type=float, default=1.0, help="IS rho clip (when offpolicy_correction != none)")
    p.add_argument("--kl_refresh_threshold", type=float, default=0.02, help="Recollect if KL(target||behavior) exceeds this")
    p.add_argument("--ess_refresh_ratio", type=float, default=0.3, help="Recollect if ESS/T drops below this")

    # Encoder schedule
    p.add_argument("--freeze_encoder_warmup_epochs", type=int, default=2)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)

    # Validation / early stopping
    p.add_argument("--val_size", type=int, default=10)
    p.add_argument("--val_sample_size", type=int, default=3)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--disable_early_stop", action="store_true")

    # Vectorization
    p.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs for vectorized exploration")

    return p.parse_args()
