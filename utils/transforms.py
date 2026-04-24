"""Rigid transform helpers used by the UR10 error simulation modules."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rot_x(angle: float) -> np.ndarray:
    """Return a homogeneous rotation around the x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rot_z(angle: float) -> np.ndarray:
    """Return a homogeneous rotation around the z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def trans_x(distance: float) -> np.ndarray:
    """Return a homogeneous translation along the x-axis."""
    transform = np.eye(4, dtype=float)
    transform[0, 3] = distance
    return transform


def trans_z(distance: float) -> np.ndarray:
    """Return a homogeneous translation along the z-axis."""
    transform = np.eye(4, dtype=float)
    transform[2, 3] = distance
    return transform


def make_transform(
    translation: np.ndarray | list[float],
    rpy: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Build a 4x4 transform from xyz translation and roll-pitch-yaw angles."""
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    if rpy is not None:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.asarray(rpy, dtype=float)
        ).as_matrix()
    return transform


def modified_dh_transform(
    alpha_prev: float,
    a_prev: float,
    theta: float,
    d: float,
) -> np.ndarray:
    """Compute one Modified DH transform.

    Convention:
        T(i-1, i) = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
    """
    return rot_x(alpha_prev) @ trans_x(a_prev) @ rot_z(theta) @ trans_z(d)


def validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a 4x4 homogeneous transform."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous matrix.")
    if not np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1].")
    return matrix


def position_from_transform(transform: np.ndarray) -> np.ndarray:
    """Extract xyz position from a homogeneous transform."""
    return np.asarray(transform, dtype=float)[:3, 3].copy()


def relative_distance_errors(
    measured_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> np.ndarray:
    """Return pairwise relative distance errors for all i < j point pairs."""
    measured = np.asarray(measured_positions, dtype=float)
    reference = np.asarray(reference_positions, dtype=float)
    if measured.shape != reference.shape or measured.shape[1] != 3:
        raise ValueError("measured_positions and reference_positions must be N x 3.")

    errors = []
    for i in range(len(measured) - 1):
        for j in range(i + 1, len(measured)):
            measured_distance = np.linalg.norm(measured[i] - measured[j])
            reference_distance = np.linalg.norm(reference[i] - reference[j])
            errors.append(measured_distance - reference_distance)
    return np.asarray(errors, dtype=float)


def rmse(values: np.ndarray) -> float:
    """Compute root mean square error."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array))))

