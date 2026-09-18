"""Pure Pursuit lateral control for Alpamayo FLU trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurePursuitConfig:
    wheelbase_m: float = 2.875
    min_lookahead_m: float = 3.0
    max_lookahead_m: float = 15.0
    speed_lookahead_gain_s: float = 0.5
    max_steer_angle_rad: float = math.radians(35.0)


def pure_pursuit_steer(
    waypoints_flu: np.ndarray,
    speed_mps: float,
    config: PurePursuitConfig | None = None,
) -> float:
    """Return normalized CARLA steering in ``[-1, 1]``.

    Alpamayo FLU has positive y to the left, while CARLA's normalized steering
    is positive to the right.  The final minus sign is therefore intentional.
    """
    config = config or PurePursuitConfig()
    points = np.asarray(waypoints_flu, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        raise ValueError(f"waypoints_flu: expected non-empty (N, >=2), got {points.shape}")
    if not np.isfinite(points[:, :2]).all() or not math.isfinite(speed_mps):
        raise ValueError("waypoints and speed must be finite")

    lookahead = float(
        np.clip(
            config.min_lookahead_m + config.speed_lookahead_gain_s * max(speed_mps, 0.0),
            config.min_lookahead_m,
            config.max_lookahead_m,
        )
    )
    distances = np.linalg.norm(points[:, :2], axis=1)
    forward = np.flatnonzero(points[:, 0] > 0.0)
    if len(forward) == 0:
        raise ValueError("trajectory has no waypoint in front of the ego vehicle")
    beyond = forward[distances[forward] >= lookahead]
    target_index = int(beyond[0] if len(beyond) else forward[-1])
    x, y = points[target_index, :2]
    target_distance = max(float(math.hypot(x, y)), 1e-6)
    alpha = math.atan2(float(y), float(x))
    steering_angle = math.atan2(
        2.0 * config.wheelbase_m * math.sin(alpha), target_distance
    )
    normalized_carla = -steering_angle / config.max_steer_angle_rad
    return float(np.clip(normalized_carla, -1.0, 1.0))
