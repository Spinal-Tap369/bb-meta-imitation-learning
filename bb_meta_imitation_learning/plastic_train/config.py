# plastic_train/config.py

import argparse

def parse_args():
    p = argparse.ArgumentParser("BC Meta-Imitation Training (SNAIL) — plastic-only (+ optional ES/SPSA)")

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

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--nbc", type=int, default=8,
                   help="Outer updates per collected explore")
    p.add_argument("--adapt_trunc_T", type=int, default=250,
                   help="Truncate explore traj to last T steps before plastic adaptation")

    # Loss / regularization
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing for exploitation BC")
    p.add_argument("--outer_pg_coef", type=float, default=0.0,
                   help="Δ-gated outer PG term weight (0 disables)")
    p.add_argument("--delta_gate_relu", action="store_true",
                   help="Use ReLU on Δ gate (only reward positive adaptation)")

    # RL / returns
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--rew_scale", type=float, default=1.0,
                   help="Scale rewards before returns/advantage")
    p.add_argument("--rew_clip", type=float, default=10.0,
                   help="Clip rewards to [-rew_clip, rew_clip] before returns")

    # Exploration rollout reuse / off-policy guards
    p.add_argument("--explore_reuse_M", type=int, default=1,
                   help="Reuse count for the same phase-1 rollout (1=no reuse)")
    p.add_argument("--is_clip_rho", type=float, default=2.0,
                   help="IS ρ clip (cap importance weights) for monitor PG")
    p.add_argument("--kl_refresh_threshold", type=float, default=0.02,
                   help="Recollect if KL(target||behavior) exceeds this")
    p.add_argument("--ess_refresh_ratio", type=float, default=0.3,
                   help="Recollect if ESS/T drops below this")

    # Encoder schedule
    p.add_argument("--freeze_encoder_warmup_epochs", type=int, default=2)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)

    # Critic (value function) training — optional & separate
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

    # Plastic knobs
    p.add_argument("--plastic_rule", type=str, default="oja", choices=["hebb", "oja"])
    p.add_argument("--plastic_eta", type=float, default=0.1)
    p.add_argument("--plastic_learn_eta", action="store_true")
    p.add_argument("--plastic_clip_mod", type=float, default=2.0)

    # Data augmentation (optional)
    p.add_argument("--use_aug", action="store_true")
    p.add_argument("--aug_prob", type=float, default=0.5)
    p.add_argument("--aug_brightness_range", type=float, nargs=2, default=(0.9, 1.1))
    p.add_argument("--aug_contrast_range",  type=float, nargs=2, default=(0.9, 1.1))
    p.add_argument("--aug_noise_std", type=float, default=0.02)
    p.add_argument("--aug_jitter_max", type=int, default=2)
    p.add_argument("--aug_bc_prob", type=float, default=0.5)
    p.add_argument("--aug_noise_prob", type=float, default=0.25)
    p.add_argument("--aug_jitter_prob", type=float, default=0.25)

    # ----- ES / SPSA (black-box meta) -----
    p.add_argument("--es_enabled", action="store_true",
                   help="Use black-box ES/SPSA for the outer update instead of backprop")
    p.add_argument("--es_algo", type=str, default="es", choices=["es", "spsa"],
                   help="Which black-box estimator to use")
    p.add_argument("--es_sigma", type=float, default=0.02,
                   help="Perturbation std (or SPSA step size)")
    p.add_argument("--es_popsize", type=int, default=8,
                   help="Number of antithetic pairs (total evals = 2*popsize)")
    p.add_argument("--es_scope", type=str, default="policy", choices=["head", "policy", "all"],
                   help="Subset of model params to optimize with ES")
    p.add_argument("--es_clip_grad", type=float, default=1.0,
                   help="Clip norm for ES gradient before optimizer.step()")

    # Debug (optional)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_level", type=str, default="INFO")
    p.add_argument("--debug_every_batches", type=int, default=1)
    p.add_argument("--debug_tasks_per_batch", type=int, default=4)
    p.add_argument("--debug_inner_per_task", action="store_true")
    p.add_argument("--debug_mem", action="store_true")
    p.add_argument("--debug_timing", action="store_true")
    p.add_argument("--debug_shapes", action="store_true")

    return p.parse_args()
