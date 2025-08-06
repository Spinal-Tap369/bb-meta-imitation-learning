# bc_pre_trainer/utils.py

import numpy as np

SEQ_LEN = 500
PAD_ACTION = -100

LEFT_ACTION     = 0
RIGHT_ACTION    = 1
STRAIGHT_ACTION = 2

MIN_STRAIGHT_RECOVERY = 3
MIN_COLLISION_RECOVERY = 4
MIN_CORNER_STRAIGHT   = 4

TURN_OVERSAMPLE_FRACTION      = 0.1
COLLISION_OVERSAMPLE_FRACTION = 0.1
CORNER_OVERSAMPLE_FRACTION    = 0.2

STEP_REWARD      = -0.01
COLLISION_PENALTY = -0.005
COLLISION_REWARD = STEP_REWARD + COLLISION_PENALTY  # -0.015

def pad_or_truncate(seq: np.ndarray, pad_value):
    L = seq.shape[0]
    if L >= SEQ_LEN:
        return seq[:SEQ_LEN]
    pad_len = SEQ_LEN - L
    pad_shape = (pad_len, *seq.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)
