"""Coordinate transforms used by the trajectory controller.

Alpamayo trajectories use an ego-local FLU frame (x forward, y left, z up).
CARLA uses x forward, y right, z up.  A CARLA ``Pose.rotation`` maps CARLA-
local vectors into world coordinates, so crossing the boundary requires one
explicit y-axis reflection.  Keeping this conversion here prevents steering
sign fixes from being scattered through the controller and simulator code.
"""

from __future__ import annotations

import numpy as np

from acarla.types import Pose

_FLIP_Y = np.diag(np.array([1.0, -1.0, 1.0], dtype=np.float32))


def flu_points_to_world(points_flu: np.ndarray, origin_world: Pose) -> np.ndarray:
    """Transform ``(N, 3)`` Alpamayo FLU points to CARLA world coordinates."""
    points = np.asarray(points_flu, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_flu: expected shape (N, 3), got {points.shape}")
    carla_local = points @ _FLIP_Y.T
    return (carla_local @ origin_world.rotation.T + origin_world.translation).astype(
        np.float32
    )


def world_points_to_flu(points_world: np.ndarray, ego_world: Pose) -> np.ndarray:
    """Transform ``(N, 3)`` CARLA world points to the current ego FLU frame."""
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world: expected shape (N, 3), got {points.shape}")
    carla_local = (points - ego_world.translation) @ ego_world.rotation
    return (carla_local @ _FLIP_Y.T).astype(np.float32)
