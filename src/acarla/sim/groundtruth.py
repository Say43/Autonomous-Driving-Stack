"""Ground-truth extraction from a live CARLA world for the trace recorder.

Imports `carla` -- only ever used from `.venv-sim` (see
`scripts/run_open_loop.py`). Not imported by anything the carla-less test
suite touches.

Coordinates are CARLA world coordinates throughout (left-handed: x forward,
y right, z up at the map origin), matching the `ActorState`/`Pose` docstring
contract in `acarla.types` -- no FLU conversion happens here; that is the
adapter's job in a later milestone, not the recorder's.
"""

from __future__ import annotations

import numpy as np

from acarla.types import ActorState, BoundingBox, LaneSegment, Pose, TrafficLightState

DEFAULT_RADIUS_M = 60.0
_ROTATION_ORTHO_TOL = 1e-4


def transform_to_pose(transform: carla.Transform) -> Pose:  # noqa: F821
    """Convert a `carla.Transform` to `acarla.types.Pose`.

    Uses `Transform.get_matrix()` (per project instructions) rather than
    hand-rolling Euler-angle trigonometry. CARLA composes this matrix from
    independent roll/pitch/yaw rotations, so it is orthonormal with
    det == +1 to floating-point precision in practice; we still defensively
    re-orthonormalize via SVD and check the determinant, and raise loudly
    if CARLA ever hands back something that isn't a re-orthonormalizable
    rotation (e.g. a genuinely scaled or reflected matrix), since silently
    "fixing" that would hide a real bug.
    """
    m = np.asarray(transform.get_matrix(), dtype=np.float64)
    translation = m[:3, 3].astype(np.float32)
    rot = m[:3, :3]

    u, _, vt = np.linalg.svd(rot)
    rot_ortho = u @ vt
    if np.linalg.det(rot_ortho) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rot_ortho = u @ vt

    deviation = np.max(np.abs(rot_ortho - rot))
    if deviation > 1e-2:
        raise ValueError(
            f"transform_to_pose: carla.Transform.get_matrix() rotation block "
            f"deviates from its nearest orthonormal matrix by {deviation}, "
            f"which is far beyond floating-point noise -- this looks like a "
            f"scaled or reflected transform, not normal CARLA output"
        )

    return Pose(translation=translation, rotation=rot_ortho.astype(np.float32))


def _actor_state(actor: carla.Actor) -> ActorState:  # noqa: F821
    bb = actor.bounding_box
    velocity = actor.get_velocity()
    return ActorState(
        id=actor.id,
        type_id=actor.type_id,
        bounding_box=BoundingBox(
            extent=np.array([bb.extent.x, bb.extent.y, bb.extent.z], dtype=np.float32),
            location=np.array([bb.location.x, bb.location.y, bb.location.z], dtype=np.float32),
        ),
        transform=transform_to_pose(actor.get_transform()),
        velocity=np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32),
    )


def actor_states(
    world: carla.World,  # noqa: F821
    ego_id: int,
    radius_m: float = DEFAULT_RADIUS_M,
) -> list[ActorState]:
    """Vehicles and pedestrians within `radius_m` of the ego actor, excluding
    the ego itself."""
    ego = world.get_actor(ego_id)
    ego_loc = ego.get_transform().location

    out: list[ActorState] = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_id:
            continue
        if actor.get_transform().location.distance(ego_loc) > radius_m:
            continue
        out.append(_actor_state(actor))
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        if actor.get_transform().location.distance(ego_loc) > radius_m:
            continue
        out.append(_actor_state(actor))
    return out


def traffic_light_states(
    world: carla.World,  # noqa: F821
    ego_location: carla.Location,  # noqa: F821
    radius_m: float = DEFAULT_RADIUS_M,
) -> list[TrafficLightState]:
    """Traffic lights within `radius_m` of `ego_location`."""
    out: list[TrafficLightState] = []
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        loc = tl.get_transform().location
        if loc.distance(ego_location) > radius_m:
            continue
        out.append(
            TrafficLightState(
                id=tl.id,
                state=str(tl.get_state()).rsplit(".", maxsplit=1)[-1].lower(),
                position=np.array([loc.x, loc.y, loc.z], dtype=np.float32),
            )
        )
    return out


def lane_segments(
    carla_map: carla.Map,  # noqa: F821
    ego_location: carla.Location,  # noqa: F821
    radius_m: float = DEFAULT_RADIUS_M,
    waypoint_distance_m: float = 2.0,
    waypoints: list | None = None,
) -> list[LaneSegment]:
    """Lane centerline polylines near `ego_location`.

    Built from `Map.generate_waypoints(waypoint_distance_m)`: waypoints
    within `radius_m` are grouped by `(road_id, section_id, lane_id)` and ordered along
    each lane's `s` (arc-length) coordinate to form a polyline. Groups with
    fewer than 2 points (can't form a polyline; also rejected by
    `LaneSegment.__post_init__`) are dropped.

    `LaneSegment.lane_id` is a bare int in the shared contract, but CARLA's
    own `lane_id` is only unique *within* a road (it repeats across roads).
    We encode `(road_id, section_id, lane_id)` into a single int deterministically;
    this is not CARLA's raw lane_id and must not be compared against it.
    """
    if waypoints is None:
        waypoints = carla_map.generate_waypoints(waypoint_distance_m)

    groups: dict[tuple[int, int, int], list] = {}
    for wp in waypoints:
        loc = wp.transform.location
        if loc.distance(ego_location) > radius_m:
            continue
        key = (wp.road_id, wp.section_id, wp.lane_id)
        groups.setdefault(key, []).append(wp)

    segments: list[LaneSegment] = []
    for (road_id, section_id, lane_id), wps in groups.items():
        if len(wps) < 2:
            continue
        wps.sort(key=lambda w: w.s)
        pts = np.array(
            [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in wps],
            dtype=np.float32,
        )
        encoded_id = (road_id * 1000 + section_id) * 1000 + (lane_id + 500)
        # A radius crop can leave disconnected pieces of a looping road.
        breaks = np.flatnonzero(np.linalg.norm(np.diff(pts, axis=0), axis=1) > 6.0) + 1
        for part in np.split(pts, breaks):
            if len(part) >= 2:
                segments.append(
                    LaneSegment(
                        lane_id=encoded_id,
                        polyline=part,
                        width_m=float(np.median([w.lane_width for w in wps])),
                    )
                )
    return segments


class SceneCache:
    """One map snapshot, including static meshes absent from world.get_actors().

    EnvironmentObject bounding boxes already use WORLD coordinates (CARLA API).
    Applying obj.transform a second time would displace parked vehicles/buildings.
    Static boxes are recorded once in the header, not duplicated on every tick.
    """

    def __init__(self, world):
        self.map = world.get_map()
        self.waypoints = self.map.generate_waypoints(2.0)
        self.objects: list[ActorState] = []
        surface_tags = {"none", "roads", "roadlines", "sidewalks", "ground", "terrain", "sky"}
        for obj in sorted(world.get_environment_objects(), key=lambda o: o.id):
            tag = str(obj.type).rsplit(".", 1)[-1].lower()
            if tag in surface_tags:
                continue
            bb = obj.bounding_box
            extent = np.array([bb.extent.x, bb.extent.y, bb.extent.z], np.float32)
            if not np.isfinite(extent).all() or np.any(extent <= 0):
                continue
            axes = [
                bb.rotation.get_forward_vector(),
                bb.rotation.get_right_vector(),
                bb.rotation.get_up_vector(),
            ]
            rotation = np.array([[a.x, a.y, a.z] for a in axes], np.float32).T
            self.objects.append(
                ActorState(
                    id=-(len(self.objects) + 1),
                    type_id=f"static.{tag}",
                    transform=Pose(
                        np.array([bb.location.x, bb.location.y, bb.location.z], np.float32),
                        rotation,
                    ),
                    bounding_box=BoundingBox(extent, np.zeros(3, np.float32)),
                    velocity=np.zeros(3, np.float32),
                )
            )
        self.centers = np.array([o.transform.translation for o in self.objects]).reshape(-1, 3)
        self.radii = np.array([np.linalg.norm(o.bounding_box.extent[:2]) for o in self.objects])

    def nearby(self, position: np.ndarray, radius_m: float = DEFAULT_RADIUS_M):
        distances = np.linalg.norm(self.centers[:, :2] - position[:2], axis=1)
        return [self.objects[i] for i in np.flatnonzero(distances <= radius_m + self.radii)]

    def lanes(self, location, radius_m: float = DEFAULT_RADIUS_M):
        return lane_segments(self.map, location, radius_m, waypoints=self.waypoints)


LANE_EDGE_TOLERANCE_M = 0.5
JUNCTION_EXTRA_TOLERANCE_M = 1.5


def make_drivable_check(
    carla_map,
    tolerance_m: float = LANE_EDGE_TOLERANCE_M,
    junction_extra_m: float = JUNCTION_EXTRA_TOLERANCE_M,
):
    """Point test used by the supervisor and the off-road metrics.

    True inside a driving-lane polygon, or within half a lane width plus
    `tolerance_m` of the nearest driving lane. The strict polygon test alone
    deadlocked closed-loop run 7 at t=32 s: the ego stood 0.6 m right of the lane
    centre, one footprint edge point lay 1 cm outside the lane line with no
    shoulder polygon there, and the supervisor held the brake for good. The
    tolerance also bridges small gaps between lane polygons. Inside junctions
    the tolerance grows by `junction_extra_m`: CARLA models a junction as
    separate turning-lane polygons with unpaved-looking gaps between them,
    although the whole interior is road surface; run 7 (third attempt) stalled
    at a green light because the ego crossed the junction 1 m left of its
    lane's centre line and the left-front corner had no polygon underneath.
    """
    import carla  # noqa: F811 -- sim-only module

    def is_drivable(point) -> bool:
        loc = carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        strict = carla_map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        if strict is not None:
            return True
        wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return False
        tol = tolerance_m + (junction_extra_m if wp.is_junction else 0.0)
        return wp.transform.location.distance(loc) <= 0.5 * wp.lane_width + tol

    return is_drivable
