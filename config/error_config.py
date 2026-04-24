"""Truth-model error distribution configuration for UR10 sim-to-real studies.

These distributions describe how a simulated physical robot deviates from the
nominal model. They are prior ranges for Monte Carlo truth generation and
sim-to-real range alignment, not direct optimization variables.
"""

from __future__ import annotations

ERROR_CONFIG = {
    "base_calibration": {
        "delta_base_xyz": {
            "distribution": "normal",
            "mean": [0.0, 0.0, 0.0],
            "std": [3.0e-4, 3.0e-4, 3.0e-4],
            "unit": "m",
        },
        "delta_base_rpy": {
            "distribution": "normal",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0e-4, 1.0e-4, 1.0e-4],
            "unit": "rad",
        },
    },
    "target_ball": {
        "delta_tool_xyz": {
            "distribution": "normal",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0e-4, 1.0e-4, 1.0e-4],
            "unit": "m",
        },
        "delta_tool_rpy": {
            "distribution": "normal",
            "mean": [0.0, 0.0, 0.0],
            "std": [5.0e-5, 5.0e-5, 5.0e-5],
            "unit": "rad",
        },
    },
    "geometric": {
        "delta_a": {
            "distribution": "normal",
            "mean": [0.0] * 6,
            "std": [2.0e-4] * 6,
            "unit": "m",
        },
        "delta_d": {
            "distribution": "normal",
            "mean": [0.0] * 6,
            "std": [2.0e-4] * 6,
            "unit": "m",
        },
        "delta_alpha": {
            "distribution": "normal",
            "mean": [0.0] * 6,
            "std": [1.0e-4] * 6,
            "unit": "rad",
        },
        "delta_theta_offset": {
            "distribution": "normal",
            "mean": [0.0] * 6,
            "std": [1.0e-4] * 6,
            "unit": "rad",
        },
    },
    "joint_flexibility": {
        # Simplified compliance term:
        #     q_eff[i] = q[i] + flex_coefficients[i] * sin(q[i])
        "flex_coefficients": {
            "distribution": "normal",
            "mean": [0.0] * 6,
            "std": [5.0e-5] * 6,
            "unit": "rad",
        },
    },
    "measurement": {
        "laser_tracker": {
            # sigma = A + B * d, d is the tracker-to-target distance in meters.
            "a": 15.0e-6,
            "b": 6.0e-6,
            "tracker_position": [0.0, 0.0, 0.0],
            "unit": "m",
        },
    },
}
