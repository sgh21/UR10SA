"""UR10 nominal and error-injected robot model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from config import error_config as default_error_config
from config import nominal_config as default_nominal_config
from utils.transforms import make_transform, modified_dh_transform, validate_transform


@dataclass
class RobotParameters:
    """Concrete numerical parameters used by FK."""

    base_xyz: np.ndarray
    base_rpy: np.ndarray
    tool_xyz: np.ndarray
    tool_rpy: np.ndarray
    alpha: np.ndarray
    a: np.ndarray
    d: np.ndarray
    theta_offset: np.ndarray
    base_transform: np.ndarray
    tool_transform: np.ndarray
    flex_coefficients: np.ndarray

    def copy(self) -> "RobotParameters":
        """Deep-copy all numerical arrays."""
        return RobotParameters(
            base_xyz=self.base_xyz.copy(),
            base_rpy=self.base_rpy.copy(),
            tool_xyz=self.tool_xyz.copy(),
            tool_rpy=self.tool_rpy.copy(),
            alpha=self.alpha.copy(),
            a=self.a.copy(),
            d=self.d.copy(),
            theta_offset=self.theta_offset.copy(),
            base_transform=self.base_transform.copy(),
            tool_transform=self.tool_transform.copy(),
            flex_coefficients=self.flex_coefficients.copy(),
        )


@dataclass
class _ParameterRef:
    """Writable reference to one scalar nominal parameter."""

    label: str
    array: np.ndarray
    index: int


class RobotModel:
    """UR10 model with Monte Carlo parameter-error injection and MD-H FK."""

    def __init__(
        self,
        nominal_module: ModuleType = default_nominal_config,
        error_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.nominal_module = nominal_module
        self.error_config = deepcopy(
            error_config if error_config is not None else default_error_config.ERROR_CONFIG
        )
        self.rng = np.random.default_rng(seed)

        self.nominal_params = self._load_nominal_params(nominal_module)
        self.true_params = self.nominal_params.copy()
        self.last_error_sample: dict[str, Any] | None = None

    @staticmethod
    def _load_nominal_params(nominal_module: ModuleType) -> RobotParameters:
        mdh = nominal_module.UR10_MDH_PARAMS
        base_xyz = RobotModel._read_nominal_value(nominal_module.BASE_POSE["xyz"], 3)
        base_rpy = RobotModel._read_nominal_value(nominal_module.BASE_POSE["rpy"], 3)
        tool_xyz = RobotModel._read_nominal_value(
            nominal_module.TARGET_BALL_POSE["xyz"], 3
        )
        tool_rpy = RobotModel._read_nominal_value(
            nominal_module.TARGET_BALL_POSE["rpy"], 3
        )
        alpha = RobotModel._read_nominal_value(mdh["alpha"], 6)
        a = RobotModel._read_nominal_value(mdh["a"], 6)
        d = RobotModel._read_nominal_value(mdh["d"], 6)
        theta_offset = RobotModel._read_nominal_value(mdh["theta_offset"], 6)
        flex_coefficients = RobotModel._read_nominal_value(
            mdh.get("flex_coefficients", [0.0] * 6),
            6,
        )

        return RobotParameters(
            base_xyz=base_xyz,
            base_rpy=base_rpy,
            tool_xyz=tool_xyz,
            tool_rpy=tool_rpy,
            alpha=alpha,
            a=a,
            d=d,
            theta_offset=theta_offset,
            base_transform=validate_transform(
                nominal_module.BASE_TRANSFORM, "BASE_TRANSFORM"
            ),
            tool_transform=validate_transform(
                nominal_module.TARGET_BALL_TRANSFORM,
                "TARGET_BALL_TRANSFORM",
            ),
            flex_coefficients=flex_coefficients,
        )

    @staticmethod
    def _read_nominal_value(item: Any, size: int) -> np.ndarray:
        value = item["value"] if isinstance(item, dict) else item
        return np.asarray(value, dtype=float).reshape(size)

    def sample_error_parameters(
        self,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        """Sample all configured parameter errors.

        ``scale`` multiplies standard deviations for normal distributions and
        half-widths for uniform distributions. It is useful for sim-to-real
        range tuning without changing the nominal distribution center.
        """
        sampled: dict[str, Any] = {}
        for group_name in (
            "base_calibration",
            "target_ball",
            "geometric",
            "joint_flexibility",
        ):
            sampled[group_name] = {}
            group = self.error_config.get(group_name, {})
            for param_name, spec in group.items():
                sampled[group_name][param_name] = self._sample_distribution(
                    spec, scale=scale
                )
        return sampled

    def inject_errors(
        self,
        error_sample: dict[str, Any] | None = None,
        scale: float = 1.0,
    ) -> RobotParameters:
        """Create a physical-world 'truth' model from sampled errors."""
        if error_sample is None:
            error_sample = self.sample_error_parameters(scale=scale)

        params = self.nominal_params.copy()

        base_error = error_sample.get("base_calibration", {})
        params.base_xyz = params.base_xyz + np.asarray(
            base_error.get("delta_base_xyz", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        params.base_rpy = params.base_rpy + np.asarray(
            base_error.get("delta_base_rpy", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        params.base_transform = params.base_transform @ make_transform(
            base_error.get("delta_base_xyz", [0.0, 0.0, 0.0]),
            base_error.get("delta_base_rpy", [0.0, 0.0, 0.0]),
        )

        target_error = error_sample.get("target_ball", {})
        params.tool_xyz = params.tool_xyz + np.asarray(
            target_error.get("delta_tool_xyz", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        params.tool_rpy = params.tool_rpy + np.asarray(
            target_error.get("delta_tool_rpy", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        params.tool_transform = params.tool_transform @ make_transform(
            target_error.get("delta_tool_xyz", [0.0, 0.0, 0.0]),
            target_error.get("delta_tool_rpy", [0.0, 0.0, 0.0]),
        )

        geometric = error_sample.get("geometric", {})
        params.a += geometric.get("delta_a", np.zeros(6, dtype=float))
        params.d += geometric.get("delta_d", np.zeros(6, dtype=float))
        params.alpha += geometric.get("delta_alpha", np.zeros(6, dtype=float))
        params.theta_offset += geometric.get(
            "delta_theta_offset", np.zeros(6, dtype=float)
        )

        flexibility = error_sample.get("joint_flexibility", {})
        params.flex_coefficients = flexibility.get(
            "flex_coefficients", np.zeros(6, dtype=float)
        )

        self.true_params = params
        self.last_error_sample = deepcopy(error_sample)
        return params

    def fk(
        self,
        joint_angles: np.ndarray | list[float],
        use_true_model: bool = False,
    ) -> np.ndarray:
        """Compute full base-to-target-ball FK for one joint vector."""
        params = self.true_params if use_true_model else self.nominal_params
        q = np.asarray(joint_angles, dtype=float).reshape(6)

        # Simplified joint flexibility: compliance increases with joint load
        # surrogate sin(q). This keeps the model differentiable and cheap.
        q_eff = q + params.theta_offset + params.flex_coefficients * np.sin(q)

        transform = params.base_transform.copy()
        for joint_index in range(6):
            transform = transform @ modified_dh_transform(
                params.alpha[joint_index],
                params.a[joint_index],
                q_eff[joint_index],
                params.d[joint_index],
            )
        return transform @ params.tool_transform

    def batch_fk(
        self,
        joint_configs: np.ndarray,
        use_true_model: bool = False,
    ) -> np.ndarray:
        """Compute FK for an N x 6 joint configuration array."""
        configs = np.asarray(joint_configs, dtype=float)
        if configs.ndim != 2 or configs.shape[1] != 6:
            raise ValueError("joint_configs must be an N x 6 array.")
        return np.stack(
            [self.fk(config, use_true_model=use_true_model) for config in configs],
            axis=0,
        )

    def get_identification_parameter_names(self) -> list[str]:
        """Return scalar nominal parameter paths enabled for identification."""
        names: list[str] = []

        for pose_name in ("BASE_POSE", "TARGET_BALL_POSE"):
            pose_config = getattr(self.nominal_module, pose_name)
            for param_name, spec in pose_config.items():
                mask = self._read_optimizable_mask(spec)
                names.extend(
                    f"{pose_name}.{param_name}[{index}]"
                    for index, enabled in enumerate(mask)
                    if enabled
                )

        for param_name, spec in self.nominal_module.UR10_MDH_PARAMS.items():
            if not isinstance(spec, dict):
                continue
            mask = self._read_optimizable_mask(spec)
            names.extend(
                f"UR10_MDH_PARAMS.{param_name}[{index}]"
                for index, enabled in enumerate(mask)
                if enabled
            )
        return names

    def get_optimizable_parameter_names(self) -> list[str]:
        """Backward-compatible alias for identification parameter discovery."""
        return self.get_identification_parameter_names()

    def get_identification_vector(self) -> tuple[np.ndarray, list[str]]:
        """Pack enabled nominal parameters into a 1-D optimization vector."""
        values: list[float] = []
        labels: list[str] = []

        for parameter_ref in self._iter_identification_refs():
            values.append(float(parameter_ref.array[parameter_ref.index]))
            labels.append(parameter_ref.label)

        return np.asarray(values, dtype=float), labels

    def set_identification_vector(self, vector: np.ndarray) -> None:
        """Write an optimizer vector back into the nominal model parameters."""
        values = np.asarray(vector, dtype=float).reshape(-1)
        refs = self._iter_identification_refs()
        if values.size != len(refs):
            raise ValueError(
                f"Identification vector has {values.size} values, "
                f"expected {len(refs)}."
            )

        for value, parameter_ref in zip(values, refs):
            parameter_ref.array[parameter_ref.index] = value

        self.nominal_params.base_transform = make_transform(
            self.nominal_params.base_xyz,
            self.nominal_params.base_rpy,
        )
        self.nominal_params.tool_transform = make_transform(
            self.nominal_params.tool_xyz,
            self.nominal_params.tool_rpy,
        )

    def _iter_identification_refs(self) -> list["_ParameterRef"]:
        refs: list[_ParameterRef] = []

        for pose_name in ("BASE_POSE", "TARGET_BALL_POSE"):
            pose_config = getattr(self.nominal_module, pose_name)
            for param_name, spec in pose_config.items():
                array = self._get_named_nominal_array(f"{pose_name}.{param_name}")
                mask = self._read_optimizable_mask(spec, expected_size=array.size)
                refs.extend(
                    _ParameterRef(
                        label=f"{pose_name}.{param_name}[{index}]",
                        array=array,
                        index=index,
                    )
                    for index, enabled in enumerate(mask)
                    if enabled
                )

        for param_name, spec in self.nominal_module.UR10_MDH_PARAMS.items():
            if not isinstance(spec, dict):
                continue
            array = self._get_named_nominal_array(f"UR10_MDH_PARAMS.{param_name}")
            mask = self._read_optimizable_mask(spec, expected_size=array.size)
            refs.extend(
                _ParameterRef(
                    label=f"UR10_MDH_PARAMS.{param_name}[{index}]",
                    array=array,
                    index=index,
                )
                for index, enabled in enumerate(mask)
                if enabled
            )

        return refs

    def _get_named_nominal_array(self, name: str) -> np.ndarray:
        if name == "BASE_POSE.xyz":
            return self.nominal_params.base_xyz
        if name == "BASE_POSE.rpy":
            return self.nominal_params.base_rpy
        if name == "TARGET_BALL_POSE.xyz":
            return self.nominal_params.tool_xyz
        if name == "TARGET_BALL_POSE.rpy":
            return self.nominal_params.tool_rpy

        prefix = "UR10_MDH_PARAMS."
        if name.startswith(prefix):
            return getattr(self.nominal_params, name.removeprefix(prefix))

        raise KeyError(f"Unknown nominal parameter path: {name}")

    @staticmethod
    def _read_optimizable_mask(
        spec: dict[str, Any],
        expected_size: int | None = None,
    ) -> np.ndarray:
        value = np.asarray(spec["value"], dtype=float).reshape(-1)
        mask_config = spec.get("is_optimizable", False)
        if isinstance(mask_config, bool):
            mask = np.full(value.shape, mask_config, dtype=bool)
        else:
            mask = np.asarray(mask_config, dtype=bool).reshape(-1)

        if mask.size != value.size:
            raise ValueError(
                "is_optimizable mask size must match value size: "
                f"got {mask.size}, expected {value.size}."
            )
        if expected_size is not None and mask.size != expected_size:
            raise ValueError(
                f"Optimizable mask size {mask.size} does not match parameter "
                f"size {expected_size}."
            )
        return mask

    def _sample_distribution(self, spec: dict[str, Any], scale: float) -> np.ndarray:
        distribution = spec["distribution"].lower()
        if distribution == "normal":
            mean = np.asarray(spec["mean"], dtype=float)
            std = np.asarray(spec["std"], dtype=float) * scale
            return self.rng.normal(loc=mean, scale=std)
        if distribution == "uniform":
            low = np.asarray(spec["low"], dtype=float)
            high = np.asarray(spec["high"], dtype=float)
            center = 0.5 * (low + high)
            half_width = 0.5 * (high - low) * scale
            return self.rng.uniform(center - half_width, center + half_width)
        raise ValueError(f"Unsupported distribution: {distribution}")
