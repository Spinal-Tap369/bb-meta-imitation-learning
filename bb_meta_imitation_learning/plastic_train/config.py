# plastic_train/config.py

import argparse
import os


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

    # Logging
    p.add_argument("--log_file", type=str, default="train_debug.txt",
                   help="Path to a txt log file (all logs will also be appended here)")
    p.add_argument("--log_level", type=str, default=None,
                   help="Override root log level (DEBUG/INFO/WARNING). If not set: DEBUG if --debug else INFO.")
    p.add_argument("--log_every_step", action="store_true",
                   help="Log [STEP ...] and [ES STEP ...] lines for every step regardless of debug_every_batches.")
    p.add_argument("--no_tqdm", action="store_true",
                   help="Disable tqdm progress bar (cleaner logs when piping to files).")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--nbc", type=int, default=8,
                   help="Outer updates per collected explore")
    p.add_argument("--adapt_trunc_T", type=int, default=250,
                   help="Truncate explore traj to last T steps before plastic adaptation")

    # Loss / regularization (note: outer loss is pure BC; Δ can reweight per-task losses)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing for exploitation BC")
    p.add_argument("--delta_gate_relu", action="store_true",
                   help="Use ReLU on standardized Δ if used for reweighting")
    p.add_argument("--delta_weight_mode", type=str, default="none", choices=["none", "std", "relu"],
                   help="If not 'none', weight per-task BC by standardized Δ (or its ReLU)")

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
                   help="IS ρ clip (cap importance weights)")
    p.add_argument("--kl_refresh_threshold", type=float, default=0.02,
                   help="Recollect if KL(target||behavior) exceeds this")
    p.add_argument("--ess_refresh_ratio", type=float, default=0.3,
                   help="Recollect if ESS/T drops below this")
    p.add_argument("--inner_use_is_mod", action="store_true",
                   help="When reusing inner rollouts, multiply plastic modulators by IS weights ρ_t.")

    # Encoder schedule
    p.add_argument("--freeze_encoder_warmup_epochs", type=int, default=2)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)

    # Critic (value function) training — optional & separate (for monitoring/value head aux only)
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
    p.add_argument("--es_recollect_inner", action="store_true",
                   help="For each perturbation (±), recollect inner rollout under the perturbed policy.")
    p.add_argument("--es_common_seed", action="store_true",
                   help="Use common random numbers (same env seed) for +/− reco.")
    p.add_argument("--es_use_is_inner", action="store_true",
                   help="If not recollecting, reuse cached trajectories but apply IS within inner for f±.")
    p.add_argument("--es_ess_min_ratio", type=float, default=0.1,
                   help="If per-perturbation ESS/T drops below this, drop task from fitness for that perturbation.")
    p.add_argument("--es_ranknorm", action="store_true",
                   help="Rank-normalize task fitness before averaging within a perturbation.")
    p.add_argument("--es_fitness_baseline", action="store_true",
                   help="Subtract unperturbed f(θ) from f± before gradient estimate.")
    p.add_argument("--es_reuse_eps_bank", action="store_true",
                   help="Pre-sample ES noise (on CPU) once per batch and reuse across nbc steps.")

    # Debug (optional)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_level", type=str, default="INFO")
    p.add_argument("--debug_every_batches", type=int, default=1)
    p.add_argument("--debug_tasks_per_batch", type=int, default=4)
    p.add_argument("--debug_inner_per_task", action="store_true")
    p.add_argument("--debug_mem", action="store_true")
    p.add_argument("--debug_timing", action="store_true")
    p.add_argument("--debug_shapes", action="store_true")

    # ----- ES inner PG (ephemeral fast-weights) -----
    p.add_argument("--es_inner_pg_alpha", type=float, default=0.0,
                   help="If > 0: one-step REINFORCE inner update (lr=alpha) inside ES fitness.")
    p.add_argument("--es_inner_pg_scope", type=str, default="head", choices=["head", "policy"],
                   help="Which params to update in the inner PG step.")
    p.add_argument("--es_inner_pg_use_is", action="store_true",
                   help="Apply IS weights rho_t in the inner PG loss when reusing behavior data.")

    # Data & CPU pipeline knobs
    default_threads = min(16, os.cpu_count() or 8)
    p.add_argument("--cpu_threads", type=int, default=default_threads,
                   help="torch.set_num_threads(); also exported to OMP_NUM_THREADS")
    p.add_argument("--cpu_workers", type=int, default=max(4, default_threads // 2),
                   help="Thread-pool workers for CPU demo decode/augmentation/build")
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
