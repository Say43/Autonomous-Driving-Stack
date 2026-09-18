"""Plausibility gate for model-produced trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acarla.types import SAMPLE_DT, PlanResult


@dataclass(frozen=True)
class SafetyLimits:
    """Limits mirror the audited Alpamayo unicycle action space."""

    max_abs_curvature_inv_m: float = 0.33
    max_abs_acceleration_mps2: float = 9.8
    max_speed_mps: float = 40.0
    max_step_m: float = 4.0
    tolerance: float = 1e-4


@dataclass(frozen=True)
class SafetyCheck:
    safe: bool
    reason: str
    max_abs_curvature_inv_m: float | None = None
    max_abs_acceleration_mps2: float | None = None
    max_speed_mps: float | None = None
    max_step_m: float | None = None


def _curvatures(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 3:
        return np.empty(0, dtype=np.float64)
    p0, p1, p2 = points_xy[:-2], points_xy[1:-1], points_xy[2:]
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    denominator = a * b * c
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    curvature = np.zeros_like(denominator)
    valid = denominator > 1e-9
    curvature[valid] = 2.0 * cross[valid] / denominator[valid]
    return curvature


def check_trajectory(plan: PlanResult, limits: SafetyLimits | None = None) -> SafetyCheck:
    """Reject non-finite or dynamically implausible Alpamayo trajectories."""
    limits = limits or SafetyLimits()
    xyz = np.asarray(plan.waypoints_xyz, dtype=np.float64)
    if not np.isfinite(xyz).all():
        return SafetyCheck(False, "trajectory contains non-finite coordinates")

    # Alpamayo's first point is t0+0.1 s, so include the ego origin when
    # deriving per-step speed and acceleration.
    xy = np.concatenate((np.zeros((1, 2), dtype=np.float64), xyz[:, :2]), axis=0)
    step_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speeds = step_lengths / SAMPLE_DT
    accelerations = np.diff(speeds) / SAMPLE_DT
    curvatures = _curvatures(xy)

    max_step = float(step_lengths.max(initial=0.0))
    max_speed = float(speeds.max(initial=0.0))
    max_acceleration = float(np.abs(accelerations).max(initial=0.0))
    max_curvature = float(np.abs(curvatures).max(initial=0.0))
    values = {
        "max_abs_curvature_inv_m": max_curvature,
        "max_abs_acceleration_mps2": max_acceleration,
        "max_speed_mps": max_speed,
        "max_step_m": max_step,
    }

    checks = (
        (max_step, limits.max_step_m, "step length"),
        (max_speed, limits.max_speed_mps, "speed"),
        (max_acceleration, limits.max_abs_acceleration_mps2, "acceleration"),
        (max_curvature, limits.max_abs_curvature_inv_m, "curvature"),
    )
    for actual, limit, label in checks:
        if not math.isfinite(actual) or actual > limit + limits.tolerance:
            return SafetyCheck(
                False,
                f"{label} {actual:.6g} exceeds limit {limit:.6g}",
                **values,
            )
    return SafetyCheck(True, "ok", **values)
