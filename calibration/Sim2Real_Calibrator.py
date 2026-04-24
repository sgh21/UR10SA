"""Sim-to-real range calibration based on relative distance RMSE alignment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from config.error_config import ERROR_CONFIG
from models.RobotModel import RobotModel
from simulation.MeasurementSim import MeasurementSimulator
from utils.transforms import position_from_transform, relative_distance_errors, rmse


class Sim2RealCalibrator:
    """Tune simulation error scale so relative-distance RMSE matches real data."""

    def __init__(
        self,
        base_error_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_error_config = deepcopy(
            base_error_config if base_error_config is not None else ERROR_CONFIG
        )
        self.seed = seed

    def evaluate_scale(
        self,
        joint_configs: np.ndarray,
        target_real_rmse: float,
        scale: float,
        monte_carlo_runs: int = 100,
    ) -> dict[str, float]:
        """Estimate simulated relative-distance RMSE under one error scale."""
        simulated_rmses = self._simulate_rmses(
            joint_configs=joint_configs,
            scale=scale,
            monte_carlo_runs=monte_carlo_runs,
        )
        mean_rmse = float(np.mean(simulated_rmses))
        return {
            "scale": float(scale),
            "simulated_rmse_mean": mean_rmse,
            "simulated_rmse_std": float(np.std(simulated_rmses)),
            "target_real_rmse": float(target_real_rmse),
            "absolute_gap": abs(mean_rmse - target_real_rmse),
        }

    def fit_error_scale(
        self,
        joint_configs: np.ndarray,
        target_real_rmse: float,
        scale_bounds: tuple[float, float] = (0.05, 20.0),
        monte_carlo_runs: int = 100,
    ) -> dict[str, float]:
        """Search for the scale that aligns simulated and real RMSE."""

        def objective(scale: float) -> float:
            result = self.evaluate_scale(
                joint_configs=joint_configs,
                target_real_rmse=target_real_rmse,
                scale=scale,
                monte_carlo_runs=monte_carlo_runs,
            )
            return result["absolute_gap"]

        optimization = minimize_scalar(
            objective,
            bounds=scale_bounds,
            method="bounded",
            options={"xatol": 1.0e-2},
        )
        return self.evaluate_scale(
            joint_configs=joint_configs,
            target_real_rmse=target_real_rmse,
            scale=float(optimization.x),
            monte_carlo_runs=monte_carlo_runs,
        )

    def scaled_error_config(self, scale: float) -> dict[str, Any]:
        """Return a config copy with truth-error ranges/stds multiplied."""
        updated = deepcopy(self.base_error_config)
        for group_name in (
            "base_calibration",
            "target_ball",
            "geometric",
            "joint_flexibility",
        ):
            for spec in updated.get(group_name, {}).values():
                if spec["distribution"] == "normal":
                    spec["std"] = (np.asarray(spec["std"], dtype=float) * scale).tolist()
                elif spec["distribution"] == "uniform":
                    low = np.asarray(spec["low"], dtype=float)
                    high = np.asarray(spec["high"], dtype=float)
                    center = 0.5 * (low + high)
                    half_width = 0.5 * (high - low) * scale
                    spec["low"] = (center - half_width).tolist()
                    spec["high"] = (center + half_width).tolist()
        return updated

    def write_error_config(
        self,
        output_path: str | Path,
        scale: float,
    ) -> Path:
        """Write the scaled config to a Python module."""
        output_path = Path(output_path)
        updated = self.scaled_error_config(scale)
        body = (
            '"""Auto-generated calibrated error configuration."""\n\n'
            "from __future__ import annotations\n\n"
            f"ERROR_CONFIG = {repr(updated)}\n"
        )
        output_path.write_text(body, encoding="utf-8")
        return output_path

    def _simulate_rmses(
        self,
        joint_configs: np.ndarray,
        scale: float,
        monte_carlo_runs: int,
    ) -> np.ndarray:
        configs = np.asarray(joint_configs, dtype=float)
        nominal_model = RobotModel(error_config=self.base_error_config, seed=self.seed)
        nominal_poses = nominal_model.batch_fk(configs, use_true_model=False)
        nominal_positions = np.stack(
            [position_from_transform(pose) for pose in nominal_poses],
            axis=0,
        )

        values = []
        seed_sequence = np.random.SeedSequence(self.seed)
        child_seeds = seed_sequence.spawn(monte_carlo_runs)
        for run_index in range(monte_carlo_runs):
            run_seed = int(child_seeds[run_index].generate_state(1)[0])
            model = RobotModel(error_config=self.base_error_config, seed=run_seed)
            model.inject_errors(scale=scale)
            simulator = MeasurementSimulator(model, seed=run_seed + 1)
            measurement = simulator.simulate(configs, use_true_model=True)
            errors = relative_distance_errors(
                measurement["measured_positions"],
                nominal_positions,
            )
            values.append(rmse(errors))
        return np.asarray(values, dtype=float)
