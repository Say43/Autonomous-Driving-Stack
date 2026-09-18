"""Conservative simulator-only intervention; never a claim about model perception.

Checks both the proposed path and the path implied by the actual steering command.
It brakes; it does not invent a route or silently replace Alpamayo with map driving.
Bounding boxes approximate geometry and can produce conservative false positives.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from acarla.types import ActorState, BoundingBox, ControlCommand, Pose


@dataclass(frozen=True)
class SupervisorConfig:
    decel_mps2: float = 4.0
    reaction_s: float = 0.5
    margin_m: float = 1.0
    sample_m: float = 0.4
    min_preview_m: float = 4.0
    ground_clearance_m: float = 0.15
    wheelbase_m: float = 2.875
    max_steer_angle_rad: float = math.radians(35)
    overhead_tags: tuple[str, ...] = ("poles", "trafficlight", "trafficsigns", "vegetation")
    overhead_min_height_m: float = 2.5
    overhead_min_span_m: float = 1.0
    """A tall pole/tree whose box is also wide is a mast with an arm or a trunk
    with a canopy: the box spans the lane at head height while the only solid
    part stands at the kerb (run 7 deadlocked on a 3.8 m x 7.8 m street light
    box 2.4 m ahead). Such boxes are skipped; kerb masts are still covered by
    the driving-lane containment test."""
    rear_axle_in_actor: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SupervisorDecision:
    brake: bool = False
    reason: str = "clear"
    hazard_distance_m: float | None = None
    object_id: int | None = None


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


FOOTPRINT_POINTS = ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (1, 0))


def footprint_is_drivable(
    ego: Pose, ego_box: BoundingBox, is_drivable: Callable[[np.ndarray], bool]
) -> bool:
    """Measured footprint metric, including when interventions are disabled."""
    for x, y in FOOTPRINT_POINTS:
        local = ego_box.location.astype(float).copy()
        local[:2] += ego_box.extent[:2] * np.array([x, y])
        local[2] = 0.0
        if not is_drivable(ego.translation + ego.rotation @ local):
            return False
    return True


def boxes_overlap(c1, r1, e1, c2, r2, e2) -> bool:
    """Separating-axis test, unlike axis-aligned distance this handles crossing cars."""
    delta = c2 - c1
    for axis in (r1[:, 0], r1[:, 1], r2[:, 0], r2[:, 1]):
        radius1 = np.abs(axis @ r1) @ e1
        radius2 = np.abs(axis @ r2) @ e2
        if abs(delta @ axis) > radius1 + radius2:
            return False
    return True


def _path_samples(ego: Pose, command: ControlCommand, world_path, horizon, cfg, world_yaws=None):
    origin = ego.translation.astype(float)
    yaw = math.atan2(ego.rotation[1, 0], ego.rotation[0, 0])
    # Roll the bicycle at its rear axle, then recover the actor origin whose
    # bounding box is tested below. A shifted rig is NOT the physical pivot.
    rear_offset = np.asarray(cfg.rear_axle_in_actor, dtype=float)[:2]
    rear_origin = origin[:2] + _rotation(yaw) @ rear_offset
    distances = np.arange(0, horizon + cfg.sample_m, cfg.sample_m)
    curvature = math.tan(command.steer * cfg.max_steer_angle_rad) / cfg.wheelbase_m
    # Bicycle rollout for the applied steering catches tracking drift as well as bad plans.
    for distance in distances:
        angle = yaw + curvature * distance
        if abs(curvature) < 1e-7:
            xy = rear_origin + distance * np.array([math.cos(yaw), math.sin(yaw)])
        else:
            xy = (
                rear_origin
                + np.array([math.sin(angle) - math.sin(yaw), math.cos(yaw) - math.cos(angle)])
                / curvature
            )
        xy = xy - _rotation(angle) @ rear_offset
        yield distance, np.array([*xy, origin[2]]), angle, "steering"
    if world_path is None or len(world_path) < 2:
        return
    points = np.asarray(world_path, dtype=float)
    # Project onto segments, not just the nearest vertex (important at fast sample spacing).
    segments = np.diff(points[:, :2], axis=0)
    fractions = np.clip(
        np.einsum("ij,ij->i", origin[:2] - points[:-1, :2], segments)
        / np.maximum(np.sum(segments**2, axis=1), 1e-9),
        0,
        1,
    )
    nearest = points[:-1, :2] + fractions[:, None] * segments
    idx = int(np.argmin(np.linalg.norm(nearest - origin[:2], axis=1)))
    start = points[idx] + fractions[idx] * (points[idx + 1] - points[idx])
    path = np.concatenate(([origin], [start], points[idx + 1 :]))
    lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    arcs = np.concatenate(([0.0], np.cumsum(lengths)))
    headings = None
    if world_yaws is not None:
        predicted = np.unwrap(np.asarray(world_yaws, dtype=float))
        if predicted.shape != (len(points),) or not np.isfinite(predicted).all():
            raise ValueError("world_yaws must match world_path and be finite")
        start_yaw = predicted[idx] + fractions[idx] * (predicted[idx + 1] - predicted[idx])
        headings = np.unwrap(np.concatenate(([yaw, start_yaw], predicted[idx + 1 :])))
    for distance in distances[1:]:
        if distance > arcs[-1]:
            break
        j = min(np.searchsorted(arcs, distance, side="right") - 1, len(lengths) - 1)
        if lengths[j] < 1e-8:
            continue
        point = path[j] + (distance - arcs[j]) / lengths[j] * (path[j + 1] - path[j])
        delta = path[j + 1] - path[j]
        body_yaw = (
            math.atan2(delta[1], delta[0])
            if headings is None
            else float(np.interp(distance, arcs, headings))
        )
        yield distance, point, body_yaw, "plan"


def check_environment(
    ego: Pose,
    speed_mps: float,
    command: ControlCommand,
    world_path: np.ndarray | None,
    objects: list[ActorState],
    is_drivable: Callable[[np.ndarray], bool],
    ego_box: BoundingBox,
    config: SupervisorConfig | None = None,
    *,
    world_yaws: np.ndarray | None = None,
) -> SupervisorDecision:
    cfg = config or SupervisorConfig()
    if not math.isfinite(speed_mps) or speed_mps < 0:
        return SupervisorDecision(True, "invalid speed")
    horizon = max(
        cfg.min_preview_m,
        speed_mps * cfg.reaction_s + speed_mps**2 / (2 * cfg.decel_mps2) + cfg.margin_m,
    )
    extent = ego_box.extent[:2].astype(float) + 0.10
    prepared = []
    for actor in objects:
        center = (
            actor.transform.translation + actor.transform.rotation @ actor.bounding_box.location
        )
        z_radius = float(np.abs(actor.transform.rotation[2]) @ actor.bounding_box.extent)
        if (
            actor.type_id.startswith("static.")
            and center[2] + z_radius < ego.translation[2] + cfg.ground_clearance_m
        ):
            continue  # manhole covers / road decals are not bumper-height obstacles
        if (
            actor.type_id.rsplit(".", 1)[-1] in cfg.overhead_tags
            and actor.bounding_box.extent[2] >= cfg.overhead_min_height_m
            and max(actor.bounding_box.extent[:2]) >= cfg.overhead_min_span_m
        ):
            continue  # arm/canopy box, see overhead_tags
        # Reject overhead objects before performing the 2D footprint test.
        ego_z = ego.translation[2] + ego_box.location[2]
        if abs(center[2] - ego_z) > z_radius + ego_box.extent[2] + 0.1:
            continue
        if np.linalg.norm(center[:2] - ego.translation[:2]) > (
            horizon
            + np.linalg.norm(actor.bounding_box.extent[:2])
            + np.linalg.norm(extent)
            + np.linalg.norm(actor.velocity[:2]) * 3.0
        ):
            continue
        actor_yaw = math.atan2(actor.transform.rotation[1, 0], actor.transform.rotation[0, 0])
        prepared.append((actor, center[:2], _rotation(actor_yaw)))
    samples = sorted(
        _path_samples(ego, command, world_path, horizon, cfg, world_yaws), key=lambda p: p[0]
    )
    for distance, point, yaw, source in samples:
        rot = _rotation(yaw)
        center = point[:2] + rot @ ego_box.location[:2]
        # Corners AND edge midpoints avoid accepting a centre on the road with wheels outside.
        for x, y in FOOTPRINT_POINTS:
            xy = center + rot @ (extent * np.array([x, y]))
            if not is_drivable(np.array([*xy, point[2]])):
                return SupervisorDecision(
                    True, f"{source}: footprint leaves driving lanes", distance
                )
        eta = min(distance / max(speed_mps, 2.0), 3.0)
        for actor, other_center, other_rot in prepared:
            # Envelope between constant velocity and immediate hard braking. Treating every
            # moving lead as stationary at every future sample needlessly blocks car following.
            velocity = actor.velocity[:2]
            other_speed = float(np.linalg.norm(velocity))
            braking_time = min(eta, other_speed / cfg.decel_mps2)
            braking_factor = braking_time - 0.5 * cfg.decel_mps2 * braking_time**2 / max(
                other_speed, 1e-9
            )
            for travel_time in (braking_factor, eta):
                predicted = other_center + velocity * travel_time
                if boxes_overlap(
                    center, rot, extent, predicted, other_rot, actor.bounding_box.extent[:2] + 0.15
                ):
                    return SupervisorDecision(
                        True, f"{source}: obstacle {actor.type_id}", distance, actor.id
                    )
    return SupervisorDecision()
