"""Coordinate transforms used by the trajectory controller.

Alpamayo trajectories use an ego-local FLU frame (x forward, y left, z up).
CARLA uses x forward, y right, z up.  A CARLA ``Pose.rotation`` maps CARLA-
local vectors into world coordinates, so crossing the boundary requires one
explicit y-axis reflection.  Keeping this conversion here prevents steering
sign fixes from being scattered through the controller and simulator code.
"""

from __future__ import annotations

import numpy as np

from acarla.types import PlanResult, Pose

_FLIP_Y = np.diag(np.array([1.0, -1.0, 1.0], dtype=np.float32))


def offset_pose(actor_pose: Pose, offset_actor: tuple | np.ndarray) -> Pose:
    """Pose of a rigidly attached point; offset is CARLA-local, in metres.

    Apply at EVERY history sample, not after making the history ego-relative:
    different vehicle points describe different arcs during a turn.
    """
    offset = np.asarray(offset_actor, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("offset must be a finite three-vector")
    return Pose(
        (actor_pose.translation + actor_pose.rotation @ offset).astype(np.float32),
        actor_pose.rotation.copy(),
    )


def plan_reference_points(
    plan: PlanResult, model_pose_world: Pose, model_to_reference: tuple | np.ndarray
) -> np.ndarray:
    """Convert model-origin predictions to another rigid vehicle point.

    The displacement rotates with each PREDICTED orientation. Adding a fixed
    world offset is incorrect in curves. Plan rotations are FLU; poses are CARLA.
    """
    offset = np.asarray(model_to_reference, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("offset must be a finite three-vector")
    positions = flu_points_to_world(plan.waypoints_xyz, model_pose_world)
    rotations = model_pose_world.rotation @ _FLIP_Y @ plan.waypoints_rot @ _FLIP_Y
    return (positions + rotations @ offset).astype(np.float32)


def plan_world_yaws(plan: PlanResult, model_pose_world: Pose) -> np.ndarray:
    """Body headings, which differ from a shifted point's path tangents."""
    rotations = model_pose_world.rotation @ _FLIP_Y @ plan.waypoints_rot @ _FLIP_Y
    return np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])


def flu_points_to_world(points_flu: np.ndarray, origin_world: Pose) -> np.ndarray:
    """Transform ``(N, 3)`` Alpamayo FLU points to CARLA world coordinates."""
    points = np.asarray(points_flu, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_flu: expected shape (N, 3), got {points.shape}")
    carla_local = points @ _FLIP_Y.T
    return (carla_local @ origin_world.rotation.T + origin_world.translation).astype(np.float32)


def world_points_to_flu(points_world: np.ndarray, ego_world: Pose) -> np.ndarray:
    """Transform ``(N, 3)`` CARLA world points to the current ego FLU frame."""
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world: expected shape (N, 3), got {points.shape}")
    carla_local = (points - ego_world.translation) @ ego_world.rotation
    return (carla_local @ _FLIP_Y.T).astype(np.float32)
