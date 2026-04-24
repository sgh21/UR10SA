"""Nominal robot parameters for the UR10 absolute positioning simulator.

All length units are meters and all angular units are radians.
The MD-H table follows:
    T(i-1, i) = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)

``is_optimizable`` belongs here because identification changes the nominal
model parameters so the nominal model fits measured target-ball data. The flag
is a per-scalar mask, so a later identification run can optimize only one
translation component or only one joint's MD-H parameter.
"""

from __future__ import annotations

import numpy as np

from utils.transforms import make_transform


BASE_POSE = {
    # Laser tracker frame -> robot base frame nominal calibration.
    "xyz": {
        "value": [0.0, 0.0, 0.0],
        # x, y, z. Example: set [True, True, False] to keep base z fixed.
        "is_optimizable": [True, True, True],
        "unit": "m",
    },
    "rpy": {
        "value": [0.0, 0.0, 0.0],
        # roll, pitch, yaw.
        "is_optimizable": [True, True, True],
        "unit": "rad",
    },
}

TARGET_BALL_POSE = {
    # Robot flange frame -> tracked target-ball center nominal offset.
    "xyz": {
        "value": [0.0, 0.0, 0.390],
        "is_optimizable": [True, True, True],
        "unit": "m",
    },
    "rpy": {
        "value": [0.0, 0.0, 0.0],
        "is_optimizable": [False, False, False],
        "unit": "rad",
    },
}

# UR10 nominal Modified DH parameters. Replace with the plant-specific table
# if your controller or calibration pipeline uses a different convention.
UR10_MDH_PARAMS = {
    "alpha": {
        "value": [0.0, np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0],
        # alpha_0 ... alpha_5.
        "is_optimizable": [True, True, True, True, True, True],
        "unit": "rad",
    },
    "a": {
        "value": [0.0, 0.0, -0.6120, -0.5723, 0.0, 0.0],
        # a_0 ... a_5. Set individual joints to False when structurally fixed.
        "is_optimizable": [True, True, True, True, True, True],
        "unit": "m",
    },
    "d": {
        "value": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
        "is_optimizable": [True, True, True, True, True, True],
        "unit": "m",
    },
    "theta_offset": {
        "value": [0.0] * 6,
        "is_optimizable": [True, True, True, True, True, True],
        "unit": "rad",
    },
    # Joint compliance can also be identified as a nominal model parameter.
    "flex_coefficients": {
        "value": [0.0] * 6,
        "is_optimizable": [True, True, True, True, True, True],
        "unit": "rad",
    },
}

BASE_TRANSFORM = make_transform(
    BASE_POSE["xyz"]["value"],
    BASE_POSE["rpy"]["value"],
)

TARGET_BALL_TRANSFORM = make_transform(
    TARGET_BALL_POSE["xyz"]["value"],
    TARGET_BALL_POSE["rpy"]["value"],
)
