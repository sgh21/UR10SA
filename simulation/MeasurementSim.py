"""Laser tracker measurement simulation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from config.error_config import ERROR_CONFIG
from models.RobotModel import RobotModel
from utils.transforms import position_from_transform


class MeasurementSimulator:
    """Generate laser-tracker-frame target-ball measurements."""

    def __init__(
        self,
        robot_model: RobotModel,
        measurement_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.robot_model = robot_model
        self.measurement_config = deepcopy(
            measurement_config
            if measurement_config is not None
            else ERROR_CONFIG["measurement"]
        )
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        joint_configs: np.ndarray,
        use_true_model: bool = True,
    ) -> dict[str, Any]:
        """Return tracker-frame target-ball truth, noisy measurements, and sigmas."""
        true_poses = self.robot_model.batch_fk(
            joint_configs,
            use_true_model=use_true_model,
        )
        true_positions = np.stack(
            [position_from_transform(pose) for pose in true_poses],
            axis=0,
        )
        sigmas = self._laser_tracker_sigmas(true_positions)
        noisy_positions = true_positions + self.rng.normal(
            loc=0.0,
            scale=sigmas[:, None],
            size=true_positions.shape,
        )
        return {
            "frame": "laser_tracker",
            "true_poses": true_poses,
            "true_positions": true_positions,
            "measured_positions": noisy_positions,
            "measurement_sigma": sigmas,
        }

    def _laser_tracker_sigmas(self, positions: np.ndarray) -> np.ndarray:
        tracker = self.measurement_config["laser_tracker"]
        tracker_position = np.asarray(tracker["tracker_position"], dtype=float)
        distances = np.linalg.norm(positions - tracker_position[None, :], axis=1)
        return float(tracker["a"]) + float(tracker["b"]) * distances
