"""Shared data contracts between the CARLA adapter, the Alpamayo model wrapper,
the controller, and the trace recorder.

This module has no dependency outside numpy and the standard library. It does
not import `carla` or any Alpamayo package -- it only describes the shapes of
data that cross those boundaries.

Three top-level structures:

- `SensorPacket`: what the CARLA adapter hands to the model wrapper each tick.
- `PlanResult`: what the model wrapper hands back.
- `TraceFrame`: one JSONL record written by the recorder per simulation tick.
  `RunHeader` is the first line of every trace file.

All array-shaped fields are validated in `__post_init__`. Violations raise
`ValueError` with the expected and actual shape/dtype spelled out.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Module constants -- the single source of truth for fixed history/plan sizes.
# ---------------------------------------------------------------------------

N_HISTORY_FRAMES = 4
"""Number of past camera frames kept per camera in a `SensorPacket`."""

N_EGO_WAYPOINTS = 16
"""Number of past ego poses kept in a `SensorPacket`."""

N_PLAN_WAYPOINTS = 64
"""Number of future waypoints produced by the model in a `PlanResult`."""

_ROTATION_TOL = 1e-4

SAMPLE_DT = 0.1
"""Seconds between consecutive samples that the model implicitly assumes.

Audit finding (docs/alpamayo_api_findings.md, "Konsequenzen fuer den
Projektplan" #3): the delta tokenizer discards timestamps entirely
(third_party/alpamayo1.5/src/alpamayo1_5/tokenizers/delta_tokenizer.py:71)
and the model implicitly treats every step as exactly 100 ms apart.
Timestamps that do not sit on this 10 Hz grid are silently misinterpreted
by the model rather than rejected by it -- so this contract must reject
them instead.
"""

SAMPLE_DT_TOLERANCE = 1e-3
"""Allowed deviation (seconds) from `SAMPLE_DT` between consecutive samples.

See `SAMPLE_DT` docstring for why this is enforced here rather than left
to the model.
"""

ALPAMAYO_CAMERA_INDICES: dict[str, int] = {
    "cross_left": 0,
    "front_wide": 1,
    "cross_right": 2,
    "front_tele": 6,
}
"""Integer camera indices the upstream model identifies cameras by.

Audit finding (docs/alpamayo_api_findings.md, "Konsequenzen fuer den
Projektplan" #4): the model has no notion of camera *names* -- it indexes
cameras by integer and sorts by ascending index for a consistent ordering
(third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py:81-89,
198-202). These are the four default camera indices.
"""

ALPAMAYO_CAMERA_PROMPT_NAMES: dict[str, str] = {
    "cross_left": "Front left camera",
    "front_wide": "Front camera",
    "cross_right": "Front right camera",
    "front_tele": "Front telephoto camera",
}
"""Clear-text camera names used in the model prompt.

Maps the same contract keys as `ALPAMAYO_CAMERA_INDICES` to the strings
the prompt template expects (third_party/alpamayo1.5/src/alpamayo1_5/helper.py:27-35).
"""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_ndarray(
    arr: Any,
    name: str,
    expected_shape: tuple[int | None, ...],
    dtype: np.dtype | type,
) -> None:
    """Validate that `arr` is an ndarray with `expected_shape` and `dtype`.

    A `None` entry in `expected_shape` matches any size in that dimension.
    Raises `ValueError` naming the expected and actual shape/dtype on failure.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name}: expected numpy.ndarray, got {type(arr).__name__}")

    actual_shape = arr.shape
    if len(actual_shape) != len(expected_shape) or not all(
        e is None or e == a for e, a in zip(expected_shape, actual_shape, strict=True)
    ):
        expected_str = tuple("*" if e is None else e for e in expected_shape)
        raise ValueError(f"{name}: expected shape {expected_str}, got {actual_shape}")

    if arr.dtype != np.dtype(dtype):
        raise ValueError(f"{name}: expected dtype {np.dtype(dtype)}, got {arr.dtype}")


def _check_strictly_increasing(values: list[float], name: str) -> None:
    """Validate that `values` is strictly monotonically increasing.

    Enforces the "oldest frame first" convention used throughout this
    project. A violation here is the most dangerous failure mode: it means
    history and future would be silently swapped.
    """
    for i in range(1, len(values)):
        if not values[i] > values[i - 1]:
            raise ValueError(
                f"{name}: timestamps must be strictly increasing "
                f"(oldest first), got {values[i - 1]} followed by {values[i]} "
                f"at index {i}"
            )


def _check_uniform_dt(values: list[float], name: str) -> None:
    """Validate that consecutive differences in `values` all equal `SAMPLE_DT`
    within `SAMPLE_DT_TOLERANCE`.

    The model never sees these timestamps -- it assumes a fixed 100 ms step
    between samples (see `SAMPLE_DT`). A spacing that drifts from that grid
    is not detected by the model; it is silently reinterpreted as exactly
    100 ms, which corrupts the implied ego/camera motion. This check turns
    that silent misinterpretation into an explicit rejection here instead.
    """
    for i in range(1, len(values)):
        actual_dt = values[i] - values[i - 1]
        if abs(actual_dt - SAMPLE_DT) > SAMPLE_DT_TOLERANCE:
            raise ValueError(
                f"{name}: non-uniform sampling at index {i}: actual delta "
                f"{actual_dt} does not match expected delta {SAMPLE_DT} "
                f"within tolerance {SAMPLE_DT_TOLERANCE} -- the model does "
                "not see timestamps and would silently treat this step as "
                "exactly 100 ms, corrupting the implied motion"
            )


def alpamayo_camera_order(names: Iterable[str]) -> list[str]:
    """Return `names` sorted by ascending Alpamayo camera index.

    This ordering is part of the model contract, not a convenience: the
    upstream loader sorts cameras by integer index before stacking them
    (third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py:
    81-89, 198-202), so callers must not rely on their own call order.

    Raises `ValueError` naming the allowed camera names if any entry in
    `names` is not a known Alpamayo camera name.
    """
    names = list(names)
    unknown = [n for n in names if n not in ALPAMAYO_CAMERA_INDICES]
    if unknown:
        raise ValueError(
            f"alpamayo_camera_order: unknown camera name(s) {unknown}, "
            f"allowed names are {sorted(ALPAMAYO_CAMERA_INDICES)}"
        )
    return sorted(names, key=lambda n: ALPAMAYO_CAMERA_INDICES[n])


def assert_valid_rotation(m: np.ndarray, name: str) -> None:
    """Validate that `m` holds one or more valid rotation matrices.

    `m` must have shape (..., 3, 3). Each trailing 3x3 block must be
    orthonormal (`R @ R.T ≈ I`) with determinant `≈ +1`, tolerance 1e-4.
    Raises `ValueError` on failure.
    """
    if not isinstance(m, np.ndarray) or m.ndim < 2 or m.shape[-2:] != (3, 3):
        actual = m.shape if isinstance(m, np.ndarray) else type(m).__name__
        raise ValueError(f"{name}: expected shape (..., 3, 3), got {actual}")

    mats = m.reshape(-1, 3, 3)
    identity = np.eye(3, dtype=mats.dtype)
    for idx, r in enumerate(mats):
        if not np.allclose(r @ r.T, identity, atol=_ROTATION_TOL):
            raise ValueError(
                f"{name}: rotation matrix at index {idx} is not orthonormal "
                f"(R @ R.T deviates from identity by more than {_ROTATION_TOL})"
            )
        det = np.linalg.det(r)
        if abs(det - 1.0) > _ROTATION_TOL:
            raise ValueError(
                f"{name}: rotation matrix at index {idx} has determinant "
                f"{det}, expected +1 (tolerance {_ROTATION_TOL})"
            )


# ---------------------------------------------------------------------------
# JSON (de)serialization helpers -- numpy arrays <-> nested lists, dtype-preserving
# ---------------------------------------------------------------------------


def _ndarray_to_json(arr: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(arr.dtype), "data": arr.tolist()}


def _ndarray_from_json(d: dict[str, Any]) -> np.ndarray:
    return np.array(d["data"], dtype=np.dtype(d["dtype"]))


# ---------------------------------------------------------------------------
# SensorPacket -- adapter -> model input
# ---------------------------------------------------------------------------


@dataclass
class SensorPacket:
    """One model-input packet assembled by the CARLA adapter.

    `images` holds `N_HISTORY_FRAMES` frames per camera, oldest frame first.
    `ego_translation`/`ego_rotation` hold `N_EGO_WAYPOINTS` past ego poses,
    expressed in the ego frame of the most recent frame, oldest first.
    """

    frame_id: int
    sim_time: float
    images: dict[str, np.ndarray]
    image_timestamps: dict[str, list[float]]
    ego_translation: np.ndarray
    ego_rotation: np.ndarray
    ego_timestamps: list[float]
    command: str
    nav_guidance: str | None

    def __post_init__(self) -> None:
        if self.images.keys() != self.image_timestamps.keys():
            raise ValueError(
                "images and image_timestamps must have the same camera keys, "
                f"got images={sorted(self.images.keys())} "
                f"image_timestamps={sorted(self.image_timestamps.keys())}"
            )

        if not self.images:
            raise ValueError(
                "images: must be a non-empty subset of the known Alpamayo "
                f"camera names, got no cameras (allowed names are "
                f"{sorted(ALPAMAYO_CAMERA_INDICES)})"
            )
        unknown_cams = set(self.images.keys()) - set(ALPAMAYO_CAMERA_INDICES)
        if unknown_cams:
            raise ValueError(
                "images: keys must be a subset of the known Alpamayo camera "
                f"names, got unknown keys {sorted(unknown_cams)} "
                f"(allowed names are {sorted(ALPAMAYO_CAMERA_INDICES)})"
            )

        for cam, img in self.images.items():
            _check_ndarray(
                img,
                f"images[{cam!r}]",
                (N_HISTORY_FRAMES, None, None, 3),
                np.uint8,
            )
            ts = self.image_timestamps[cam]
            if len(ts) != N_HISTORY_FRAMES:
                raise ValueError(
                    f"image_timestamps[{cam!r}]: expected length {N_HISTORY_FRAMES}, got {len(ts)}"
                )
            _check_strictly_increasing(ts, f"image_timestamps[{cam!r}]")
            _check_uniform_dt(ts, f"image_timestamps[{cam!r}]")

        _check_ndarray(
            self.ego_translation,
            "ego_translation",
            (N_EGO_WAYPOINTS, 3),
            np.float32,
        )
        _check_ndarray(
            self.ego_rotation,
            "ego_rotation",
            (N_EGO_WAYPOINTS, 3, 3),
            np.float32,
        )
        assert_valid_rotation(self.ego_rotation, "ego_rotation")

        if len(self.ego_timestamps) != N_EGO_WAYPOINTS:
            raise ValueError(
                f"ego_timestamps: expected length {N_EGO_WAYPOINTS}, got {len(self.ego_timestamps)}"
            )
        _check_strictly_increasing(self.ego_timestamps, "ego_timestamps")
        _check_uniform_dt(self.ego_timestamps, "ego_timestamps")


# ---------------------------------------------------------------------------
# PlanResult -- model output
# ---------------------------------------------------------------------------


@dataclass
class PlanResult:
    """The model's planned future trajectory for one frame."""

    frame_id: int
    waypoints_xyz: np.ndarray
    waypoints_rot: np.ndarray
    reasoning: str | None
    inference_ms: float
    model_config_hash: str

    def __post_init__(self) -> None:
        _check_ndarray(
            self.waypoints_xyz,
            "waypoints_xyz",
            (N_PLAN_WAYPOINTS, 3),
            np.float32,
        )
        _check_ndarray(
            self.waypoints_rot,
            "waypoints_rot",
            (N_PLAN_WAYPOINTS, 3, 3),
            np.float32,
        )
        assert_valid_rotation(self.waypoints_rot, "waypoints_rot")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (numpy arrays -> nested lists)."""
        return {
            "frame_id": self.frame_id,
            "waypoints_xyz": _ndarray_to_json(self.waypoints_xyz),
            "waypoints_rot": _ndarray_to_json(self.waypoints_rot),
            "reasoning": self.reasoning,
            "inference_ms": self.inference_ms,
            "model_config_hash": self.model_config_hash,
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> PlanResult:
        """Deserialize from a dict produced by `to_json_dict`."""
        return PlanResult(
            frame_id=d["frame_id"],
            waypoints_xyz=_ndarray_from_json(d["waypoints_xyz"]),
            waypoints_rot=_ndarray_from_json(d["waypoints_rot"]),
            reasoning=d["reasoning"],
            inference_ms=d["inference_ms"],
            model_config_hash=d["model_config_hash"],
        )


# ---------------------------------------------------------------------------
# Small named structures used inside TraceFrame -- no loose dicts.
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """A rigid transform: translation + rotation matrix."""

    translation: np.ndarray  # (3,) float32
    rotation: np.ndarray  # (3, 3) float32

    def __post_init__(self) -> None:
        _check_ndarray(self.translation, "translation", (3,), np.float32)
        _check_ndarray(self.rotation, "rotation", (3, 3), np.float32)
        assert_valid_rotation(self.rotation, "rotation")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "translation": _ndarray_to_json(self.translation),
            "rotation": _ndarray_to_json(self.rotation),
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> Pose:
        return Pose(
            translation=_ndarray_from_json(d["translation"]),
            rotation=_ndarray_from_json(d["rotation"]),
        )


@dataclass
class BoundingBox:
    """An actor's 3D bounding box, in the actor's local frame."""

    extent: np.ndarray  # (3,) float32, half-sizes
    location: np.ndarray  # (3,) float32, offset from actor transform origin

    def __post_init__(self) -> None:
        _check_ndarray(self.extent, "extent", (3,), np.float32)
        _check_ndarray(self.location, "location", (3,), np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "extent": _ndarray_to_json(self.extent),
            "location": _ndarray_to_json(self.location),
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> BoundingBox:
        return BoundingBox(
            extent=_ndarray_from_json(d["extent"]),
            location=_ndarray_from_json(d["location"]),
        )


@dataclass
class ActorState:
    """One non-ego actor's state at a given tick, in CARLA world coordinates."""

    id: int
    type_id: str
    bounding_box: BoundingBox
    transform: Pose
    velocity: np.ndarray  # (3,) float32

    def __post_init__(self) -> None:
        _check_ndarray(self.velocity, "velocity", (3,), np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type_id": self.type_id,
            "bounding_box": self.bounding_box.to_json_dict(),
            "transform": self.transform.to_json_dict(),
            "velocity": _ndarray_to_json(self.velocity),
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> ActorState:
        return ActorState(
            id=d["id"],
            type_id=d["type_id"],
            bounding_box=BoundingBox.from_json_dict(d["bounding_box"]),
            transform=Pose.from_json_dict(d["transform"]),
            velocity=_ndarray_from_json(d["velocity"]),
        )


@dataclass
class LaneSegment:
    """One lane-geometry polyline near the ego vehicle, in world coordinates."""

    lane_id: int
    polyline: np.ndarray  # (N, 3) float32, N >= 2
    width_m: float | None = None

    def __post_init__(self) -> None:
        _check_ndarray(self.polyline, "polyline", (None, 3), np.float32)
        if self.width_m is not None and (not np.isfinite(self.width_m) or self.width_m <= 0):
            raise ValueError("lane width must be finite and positive")
        if self.polyline.shape[0] < 2:
            raise ValueError(f"polyline: expected at least 2 points, got {self.polyline.shape[0]}")

    def to_json_dict(self) -> dict[str, Any]:
        result = {"lane_id": self.lane_id, "polyline": _ndarray_to_json(self.polyline)}
        if self.width_m is not None:
            result["width_m"] = self.width_m
        return result

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> LaneSegment:
        return LaneSegment(
            lane_id=d["lane_id"],
            polyline=_ndarray_from_json(d["polyline"]),
            width_m=d.get("width_m"),
        )


@dataclass
class TrafficLightState:
    """One traffic light's state at a given tick."""

    id: int
    state: str
    position: np.ndarray  # (3,) float32

    def __post_init__(self) -> None:
        _check_ndarray(self.position, "position", (3,), np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "position": _ndarray_to_json(self.position),
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> TrafficLightState:
        return TrafficLightState(
            id=d["id"], state=d["state"], position=_ndarray_from_json(d["position"])
        )


@dataclass
class ControlCommand:
    """The control command applied at a given tick."""

    steer: float
    throttle: float
    brake: float

    def to_json_dict(self) -> dict[str, Any]:
        return {"steer": self.steer, "throttle": self.throttle, "brake": self.brake}

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> ControlCommand:
        return ControlCommand(steer=d["steer"], throttle=d["throttle"], brake=d["brake"])


# ---------------------------------------------------------------------------
# RunHeader -- first line of every JSONL trace file
# ---------------------------------------------------------------------------


@dataclass
class RunHeader:
    """Metadata written once, as the first line of a run's JSONL trace file."""

    carla_version: str
    map_name: str
    seed_world: int
    seed_traffic_manager: int
    model_config_name: str
    model_config_hash: str
    git_sha: str
    run_timestamp: str
    cameras: list[str]
    fixed_delta_seconds: float
    environment_objects: list[ActorState] | None = None
    vehicle_geometry: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        result = {
            "carla_version": self.carla_version,
            "map_name": self.map_name,
            "seed_world": self.seed_world,
            "seed_traffic_manager": self.seed_traffic_manager,
            "model_config_name": self.model_config_name,
            "model_config_hash": self.model_config_hash,
            "git_sha": self.git_sha,
            "run_timestamp": self.run_timestamp,
            "cameras": list(self.cameras),
            "fixed_delta_seconds": self.fixed_delta_seconds,
        }
        if self.environment_objects is not None:
            result["environment_objects"] = [a.to_json_dict() for a in self.environment_objects]
        if self.vehicle_geometry:
            result["vehicle_geometry"] = dict(self.vehicle_geometry)
        return result

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> RunHeader:
        return RunHeader(
            carla_version=d["carla_version"],
            map_name=d["map_name"],
            seed_world=d["seed_world"],
            seed_traffic_manager=d["seed_traffic_manager"],
            model_config_name=d["model_config_name"],
            model_config_hash=d["model_config_hash"],
            git_sha=d["git_sha"],
            run_timestamp=d["run_timestamp"],
            cameras=list(d["cameras"]),
            fixed_delta_seconds=d["fixed_delta_seconds"],
            environment_objects=(
                [ActorState.from_json_dict(a) for a in d["environment_objects"]]
                if "environment_objects" in d
                else None
            ),
            vehicle_geometry=dict(d.get("vehicle_geometry", {})),
        )


# ---------------------------------------------------------------------------
# TraceFrame -- one JSONL record per simulation tick
# ---------------------------------------------------------------------------


@dataclass
class TraceFrame:
    """One recorded simulation tick. Images are never embedded here -- only
    paths, relative to the run directory, into files written alongside the
    trace."""

    frame_id: int
    sim_time: float
    ego_pose_world: Pose
    actors: list[ActorState]
    lanes: list[LaneSegment]
    traffic_lights: list[TrafficLightState]
    plan: PlanResult | None
    control: ControlCommand
    image_paths: dict[str, list[str]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Camera-rig/model origin, distinct from the actor used to draw the vehicle.
    # Absent in legacy traces: those plans were anchored at ego_pose_world.
    model_pose_world: Pose | None = None

    @property
    def inference_pose_world(self) -> Pose:
        return self.model_pose_world if self.model_pose_world is not None else self.ego_pose_world

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (numpy arrays -> nested lists)."""
        result = {
            "frame_id": self.frame_id,
            "sim_time": self.sim_time,
            "ego_pose_world": self.ego_pose_world.to_json_dict(),
            "actors": [a.to_json_dict() for a in self.actors],
            "lanes": [lane.to_json_dict() for lane in self.lanes],
            "traffic_lights": [t.to_json_dict() for t in self.traffic_lights],
            "plan": self.plan.to_json_dict() if self.plan is not None else None,
            "control": self.control.to_json_dict(),
            "image_paths": {k: list(v) for k, v in self.image_paths.items()},
        }
        if self.diagnostics:
            result["diagnostics"] = dict(self.diagnostics)
        if self.model_pose_world is not None:
            result["model_pose_world"] = self.model_pose_world.to_json_dict()
        return result

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> TraceFrame:
        """Deserialize from a dict produced by `to_json_dict`."""
        return TraceFrame(
            frame_id=d["frame_id"],
            sim_time=d["sim_time"],
            ego_pose_world=Pose.from_json_dict(d["ego_pose_world"]),
            actors=[ActorState.from_json_dict(a) for a in d["actors"]],
            lanes=[LaneSegment.from_json_dict(lane) for lane in d["lanes"]],
            traffic_lights=[TrafficLightState.from_json_dict(t) for t in d["traffic_lights"]],
            plan=PlanResult.from_json_dict(d["plan"]) if d["plan"] is not None else None,
            control=ControlCommand.from_json_dict(d["control"]),
            image_paths={k: list(v) for k, v in d["image_paths"].items()},
            diagnostics=dict(d.get("diagnostics", {})),
            model_pose_world=(
                Pose.from_json_dict(d["model_pose_world"])
                if d.get("model_pose_world") is not None
                else None
            ),
        )
