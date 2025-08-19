# plastic_train/__init__.py

"""
BC Meta-Imitation training (SNAIL) — Vectorized exploration

Public API:
- run_training(): launches the training loop (vectorized explore)
- parse_args():   CLI arg parser used by run_training()
"""
from .train import run_training
from .config import parse_args

__all__ = ["run_training", "parse_args"]
__version__ = "0.1.0"
