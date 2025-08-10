# bc_pre_trainer/utils.py

import numpy as np

# Sequence / padding
SEQ_LEN = 500
PAD_ACTION = -100

# Action codes
LEFT_ACTION     = 0
RIGHT_ACTION    = 1
STRAIGHT_ACTION = 2

# Mining thresholds
MIN_STRAIGHT_RECOVERY = 3   # used by turn & collision recovery checks
MIN_COLLISION_RECOVERY = 4
MIN_CORNER_STRAIGHT   = 4   # number of straight steps after a corner
MIN_STRAIGHT_SEGMENT  = 4   # min length for pure-straight segments

# Oversampling mix 
TURN_OVERSAMPLE_FRACTION       = 0.05
COLLISION_OVERSAMPLE_FRACTION  = 0.05
CORNER_OVERSAMPLE_FRACTION     = 0.1
STRAIGHT_OVERSAMPLE_FRACTION   = 0.05

# Rewards (for collision mining)
STEP_REWARD       = -0.01
COLLISION_PENALTY = -0.005
COLLISION_REWARD  = STEP_REWARD + COLLISION_PENALTY  # -0.015

# Utils
def pad_or_truncate(seq: np.ndarray, pad_value):
    L = seq.shape[0]
    if L >= SEQ_LEN:
        return seq[:SEQ_LEN]
    pad_len = SEQ_LEN - L
    pad_shape = (pad_len, *seq.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)

