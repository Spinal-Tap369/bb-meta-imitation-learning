# mri_train/config.py

import argparse
import os
import torch

def parse_args():
    p = argparse.ArgumentParser(
        "BC Meta-Imitation Training — MRI (inner PG + outer BC + optional meta-descent correction)"
    )

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo_root", type=str, required=True,
                   help="Root with demo_manifest.csv, dagger_manifest.csv, and demos/")
    p.add_argument("--train_trials", type=str, required=True,
                   help="Path to train_trials.json")
    p.add_argument("--bc_init", type=str, default=None,
                   help="Optional BC pretrain checkpoint (state_dict)")
    p.add_argument("--save_path", type=str, default="bc_meta_ckpts")
    p.add_argument("--load_path", type=str, default="bc_meta_ckpts_load")

    # Logging
    p.add_argument("--log_file", type=str, default="train_debug.txt",
                   help="Path to a txt log file (all logs will also be appended here)")
    p.add_argument("--log_level", type=str, default=None,
                   help="Override root log level (DEBUG/INFO/WARNING). If not set: DEBUG if --debug else INFO.")
    p.add_argument("--no_tqdm", action="store_true",
                   help="Disable tqdm progress bar (cleaner logs when piping to files).")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_level", type=str, default="INFO")
    p.add_argument("--debug_every_batches", type=int, default=1)
    p.add_argument("--debug_tasks_per_batch", type=int, default=4)
    p.add_argument("--debug_timing", action="store_true")
    p.add_argument("--debug_shapes", action="store_true")

    # Training schedule
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--nbc", type=int, default=8,
                   help="Outer updates per collected explore rollout. "
                        "Explore reuse is fixed to this value (no mid-batch recollects).")
    p.add_argument("--adapt_trunc_T", type=int, default=250,
                   help="Truncate explore traj to last T steps before inner PG (0=disable)")

    # Loss / regularization
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing for outer BC")

    # RL / returns (inner PG)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--rew_scale", type=float, default=1.0,
                   help="Scale rewards before returns/advantage")
    p.add_argument("--rew_clip", type=float, default=10.0,
                   help="Clip rewards to [-rew_clip, rew_clip] before returns")

    # Off-policy / IS controls (no recollects; guards only)
    p.add_argument("--inner_use_is", action="store_true",
                   help="Apply per-timestep IS weights rho_t when reusing the rollout")
    p.add_argument("--inner_is_ref", type=str, default="theta_init",
                   choices=["theta_init", "beh"],
                   help="Reference policy for IS ratios: start-of-batch θ_init (recommended) or behavior logits.")
    p.add_argument("--is_clip_rho", type=float, default=2.0,
                   help="Per-timestep IS rho clip when reusing rollouts (0=no clip)")
    p.add_argument("--inner_ess_min", type=float, default=0.0,
                   help="ESS/T threshold to SKIP inner PG if too off-policy (e.g., 0.3). 0 disables gate.")

    # MRI — inner policy gradient (REINFORCE on rollout)
    p.add_argument("--inner_pg_alpha", type=float, default=0.1,
                   help="Inner PG step size α")
    p.add_argument("--inner_steps", type=int, default=1,
                   help="Number of inner PG steps")
    p.add_argument("--second_order", action="store_true",
                   help="Differentiate through inner step(s) (MAML-style). If off, uses first-order.")

    # MRI — meta-descent correction (score-function term)
    p.add_argument("--meta_corr", action="store_true",
                   help="Enable meta-descent correction term")
    p.add_argument("--meta_corr_coeff", type=float, default=1.0,
                   help="Multiplier for the correction loss")
    p.add_argument("--meta_corr_baseline", type=str, default="batch",
                   choices=["batch", "ema", "none"],
                   help="Baseline for correction term: batch mean, EMA, or none")
    p.add_argument("--meta_corr_ema_beta", type=float, default=0.9,
                   help="EMA decay for correction baseline when --meta_corr_baseline=ema")
    p.add_argument("--meta_corr_use_is", action="store_true",
                   help="Multiply log-prob sum by rho_t (detach) in correction")
    p.add_argument("--meta_corr_center_logp", action="store_true",
                   help="Center per-timestep logπ with its mean to reduce variance")

    # Encoder schedule
    p.add_argument("--freeze_encoder_warmup_epochs", type=int, default=2)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)

    # Critic (value head) aux training — optional (not used by MRI gradient)
    p.add_argument("--critic_aux_steps", type=int, default=0,
                   help="Auxiliary critic-only regression steps per batch")
    p.add_argument("--critic_lr_mult", type=float, default=1.0)
    p.add_argument("--critic_value_clip", type=float, default=0.0)

    # Validation / early stopping
    p.add_argument("--val_size", type=int, default=10)
    p.add_argument("--val_sample_size", type=int, default=3)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--disable_early_stop", action="store_true")

    # Vectorization
    p.add_argument("--num_envs", type=int, default=8,
                   help="Number of parallel envs for vectorized exploration")

    # Data & CPU pipeline knobs
    default_threads = min(16, os.cpu_count() or 8)
    p.add_argument("--cpu_threads", type=int, default=default_threads)
    p.add_argument("--cpu_workers", type=int, default=max(4, default_threads // 2))
    p.add_argument("--pin_memory", action="store_true",
                   help="Pin host memory for H2D and use a dedicated CUDA copy stream")
    p.add_argument("--prefetch_batches", type=int, default=1,
                   help="One-batch look-ahead NPZ prefetch (CPU-only). Set 0 to disable.")
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="Enable cudnn.benchmark for convs (slightly nondeterministic).")

    # Synthetic demos
    p.add_argument("--syn_demo_root", type=str, default=None,
                   help="Optional second demo root for synthetic demos (same layout as main).")
    p.add_argument("--syn_demo_min_epoch", type=int, default=3,
                   help="Begin including synthetic demos at this (1-indexed) epoch.")
    p.add_argument("--max_main_demos_per_task", type=int, default=3,
                   help="Max number of main demos per task to use.")
    p.add_argument("--max_syn_demos_per_task", type=int, default=20,
                   help="Max number of synthetic demos per task to use once enabled.")


    return p.parse_args()
