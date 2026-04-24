"""Minimal runnable example for the UR10 error simulation project."""

from __future__ import annotations

import numpy as np

from calibration.Sim2Real_Calibrator import Sim2RealCalibrator
from models.RobotModel import RobotModel
from simulation.MeasurementSim import MeasurementSimulator
from utils.transforms import relative_distance_errors, rmse


def build_demo_joint_configs() -> np.ndarray:
    """Create a small deterministic pose set in radians."""
    degrees = np.array(
        [
            [0, -90, 90, -90, -90, 0],
            [20, -80, 100, -110, -80, 15],
            [-25, -100, 80, -70, -95, -20],
            [35, -70, 70, -95, -75, 30],
            [-15, -85, 110, -120, -105, 10],
            [45, -95, 95, -80, -65, -35],
        ],
        dtype=float,
    )
    return np.deg2rad(degrees)


def main() -> None:
    joint_configs = build_demo_joint_configs()

    robot = RobotModel(seed=7)
    robot.inject_errors(scale=1.0)
    simulator = MeasurementSimulator(robot, seed=11)
    measurement = simulator.simulate(joint_configs, use_true_model=True)

    nominal_positions = robot.batch_fk(joint_configs, use_true_model=False)[:, :3, 3]
    distance_errors = relative_distance_errors(
        measurement["measured_positions"],
        nominal_positions,
    )

    print("Identification parameters:", robot.get_identification_parameter_names())
    print(f"Relative distance RMSE: {rmse(distance_errors):.8f} m")
    print("First measured point:", measurement["measured_positions"][0])

    calibrator = Sim2RealCalibrator(seed=13)
    result = calibrator.fit_error_scale(
        joint_configs=joint_configs,
        target_real_rmse=3.0e-4,
        scale_bounds=(0.1, 5.0),
        monte_carlo_runs=20,
    )
    print("Calibrated scale summary:", result)


if __name__ == "__main__":
    main()
