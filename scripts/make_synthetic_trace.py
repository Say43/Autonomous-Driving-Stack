#!/usr/bin/env python3
"""Generate a deterministic, synthetic trace.jsonl for viewer development.

This script does not depend on CARLA or on the Alpamayo model. It exists so
that `viewer/` can be built and tested against a plausible trace before M2
(the CARLA adapter) and M4 (the model wrapper) produce real logs.

The scene: the ego vehicle drives a gentle S-curve, a lead vehicle cruises
ahead, another vehicle overtakes and cuts back in, a pedestrian crosses near
a traffic light, and lane polylines run alongside the ego route. The planned
trajectory (`PlanResult`) is only recomputed every 5 ticks (2 Hz replanning)
and is held fixed between replans, so the viewer can show a controller
tracking a stale plan. `reasoning` is populated only on every third replan.

Usage:
    python scripts/make_synthetic_trace.py --out results/synthetic_run --ticks 200 --seed 0

Writes `<out>/trace.jsonl`: first line a `RunHeader`, then one `TraceFrame`
per tick, JSON Lines. Given the same seed and tick count, the output bytes
are identical across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.types import (  # noqa: E402
    N_PLAN_WAYPOINTS,
    SAMPLE_DT,
    ActorState,
    BoundingBox,
    ControlCommand,
    LaneSegment,
    PlanResult,
    Pose,
    RunHeader,
    TraceFrame,
    TrafficLightState,
)

REPLAN_INTERVAL = 5
"""Ticks between replans -- 2 Hz replanning at SAMPLE_DT = 0.1s (5 * 0.1 = 0.5s)."""

REASONING_EVERY_N_REPLANS = 3
"""Only every third replan carries a fresh reasoning trace."""

LANE_HALF_WIDTH = 1.75
"""Half the lane width in meters, used to offset the left/right lane polylines."""


def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
    """A right-handed rotation about +Z by `yaw` radians, as a (3, 3) float32
    array. This is a valid orthonormal rotation matrix with det == +1 by
    construction, satisfying the validation in `acarla.types`."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _ego_position(t: float, rng_amplitude: float, rng_freq: float) -> tuple[float, float, float]:
    """Ego drives along +X at constant speed while its Y offset traces a
    gentle sine (an S-curve). Returns (x, y, yaw)."""
    speed = 8.0  # m/s, roughly city speed
    x = speed * t
    y = rng_amplitude * np.sin(rng_freq * t)
    # yaw = heading of the path tangent
    dy_dx = rng_amplitude * rng_freq * np.cos(rng_freq * t)
    yaw = np.arctan2(dy_dx * speed, speed)
    return x, y, yaw


def _lane_polyline(
    x_values: np.ndarray, rng_amplitude: float, rng_freq: float, side: float
) -> np.ndarray:
    """Build a lane polyline offset `side * LANE_HALF_WIDTH` from the ego
    S-curve centerline, sampled at the given x values. `side` is +1 (left)
    or -1 (right)."""
    points = np.zeros((len(x_values), 3), dtype=np.float32)
    for i, x in enumerate(x_values):
        t = x / 8.0
        y_center = rng_amplitude * np.sin(rng_freq * t)
        dy_dx = rng_amplitude * rng_freq * np.cos(rng_freq * t)
        tangent = np.array([1.0, dy_dx], dtype=np.float64)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        offset = normal * side * LANE_HALF_WIDTH
        points[i, 0] = x + offset[0]
        points[i, 1] = y_center + offset[1]
        points[i, 2] = 0.0
    return points


def _make_plan(
    frame_id: int,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    rng_amplitude: float,
    rng_freq: float,
    reasoning: str | None,
    seed: int,
) -> PlanResult:
    """Build a plausible N_PLAN_WAYPOINTS trajectory in ego-local FLU.

    The S-curve is generated in CARLA world coordinates, then transformed
    into the current ego frame exactly as a real Alpamayo ``PlanResult`` is
    represented: x forward, y left, z up.
    """
    speed = 8.0
    waypoints_xyz = np.zeros((N_PLAN_WAYPOINTS, 3), dtype=np.float32)
    waypoints_rot = np.zeros((N_PLAN_WAYPOINTS, 3, 3), dtype=np.float32)
    t0 = ego_x / speed
    for i in range(N_PLAN_WAYPOINTS):
        t = t0 + (i + 1) * SAMPLE_DT
        x, y, yaw = _ego_position(t, rng_amplitude, rng_freq)
        dx = x - ego_x
        dy = y - ego_y
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        forward = dx * cos_yaw + dy * sin_yaw
        left = dx * sin_yaw - dy * cos_yaw
        waypoints_xyz[i] = (forward, left, 0.0)
        waypoints_rot[i] = _yaw_rotation_matrix(-(yaw - ego_yaw))
    return PlanResult(
        frame_id=frame_id,
        waypoints_xyz=waypoints_xyz,
        waypoints_rot=waypoints_rot,
        reasoning=reasoning,
        inference_ms=float(120 + (seed + frame_id) % 40),
        model_config_hash="synthetic-0000000000000000",
    )


def _reasoning_text(replan_index: int) -> str:
    texts = [
        "Lead vehicle holding speed ahead in the same lane; continuing to "
        "follow the current lane centerline through the upcoming curve.",
        "Pedestrian near the crosswalk to the right; maintaining lane "
        "position and current speed while monitoring the traffic light.",
        "Traffic light ahead showing a non-green state; preparing to "
        "reduce speed while the adjacent vehicle completes its overtake.",
    ]
    return texts[replan_index % len(texts)]


def _other_vehicle_state(
    frame_id: int, t: float, rng_amplitude: float, rng_freq: float
) -> tuple[float, float, float]:
    """A lead vehicle cruising a fixed distance ahead of the ego on the same
    S-curve centerline."""
    lead_offset_t = 2.0  # seconds ahead
    x, y, yaw = _ego_position(t + lead_offset_t, rng_amplitude, rng_freq)
    return x, y, yaw


def _overtaking_vehicle_state(
    t: float, rng_amplitude: float, rng_freq: float, total_duration: float
) -> tuple[float, float, float]:
    """A second vehicle that starts behind the ego in the left lane, overtakes,
    and cuts back into the ego lane ahead by the end of the run."""
    speed = 11.0
    x = speed * t - 15.0
    progress = min(max(t / max(total_duration, 1e-6), 0.0), 1.0)
    # Lateral offset: starts one lane to the left, eases back to centerline.
    lateral = LANE_HALF_WIDTH * 2.2 * (1.0 - progress)
    _, y_center, yaw = _ego_position(t, rng_amplitude, rng_freq)
    y = y_center + lateral
    return x, y, yaw


def _pedestrian_state(
    t: float, rng_amplitude: float, rng_freq: float
) -> tuple[float, float, float]:
    """A pedestrian crossing perpendicular to the road near x ~ 40m."""
    cross_x = 40.0
    _, y_center, _ = _ego_position(cross_x / 8.0, rng_amplitude, rng_freq)
    walk_speed = 1.2
    y = y_center - 6.0 + walk_speed * t
    yaw = np.pi / 2.0
    return cross_x, y, yaw


def build_frames(
    ticks: int, seed: int
) -> tuple[RunHeader, list[TraceFrame]]:
    rng = np.random.default_rng(seed)
    rng_amplitude = 3.0 + float(rng.uniform(-0.3, 0.3))
    rng_freq = 0.05 + float(rng.uniform(-0.01, 0.01))

    header = RunHeader(
        carla_version="synthetic",
        map_name="synthetic-s-curve",
        seed_world=seed,
        seed_traffic_manager=seed,
        model_config_name="synthetic-reference",
        model_config_hash="synthetic-0000000000000000",
        git_sha="0000000000000000000000000000000000000",
        run_timestamp="1970-01-01T00:00:00Z",
        cameras=["front_wide", "cross_left", "cross_right"],
        fixed_delta_seconds=SAMPLE_DT,
    )

    total_duration = ticks * SAMPLE_DT

    # Fixed actor identities.
    lead_id = 1001
    overtaker_id = 1002
    pedestrian_id = 2001
    traffic_light_id = 3001

    x_grid = np.linspace(-20.0, 8.0 * total_duration + 60.0, 200)
    left_lane_polyline = _lane_polyline(x_grid, rng_amplitude, rng_freq, side=1.0)
    right_lane_polyline = _lane_polyline(x_grid, rng_amplitude, rng_freq, side=-1.0)

    frames: list[TraceFrame] = []
    current_plan: PlanResult | None = None
    replan_index = -1

    for frame_id in range(ticks):
        t = frame_id * SAMPLE_DT
        ego_x, ego_y, ego_yaw = _ego_position(t, rng_amplitude, rng_freq)
        ego_pose = Pose(
            translation=np.array([ego_x, ego_y, 0.0], dtype=np.float32),
            rotation=_yaw_rotation_matrix(ego_yaw),
        )

        # Replan every REPLAN_INTERVAL ticks; hold fixed between replans.
        if frame_id % REPLAN_INTERVAL == 0:
            replan_index += 1
            reasoning = (
                _reasoning_text(replan_index // REASONING_EVERY_N_REPLANS)
                if replan_index % REASONING_EVERY_N_REPLANS == 0
                else None
            )
            current_plan = _make_plan(
                frame_id, ego_x, ego_y, ego_yaw, rng_amplitude, rng_freq, reasoning, seed
            )

        # Other vehicles.
        lead_x, lead_y, lead_yaw = _other_vehicle_state(frame_id, t, rng_amplitude, rng_freq)
        lead_actor = ActorState(
            id=lead_id,
            type_id="vehicle.tesla.model3",
            bounding_box=BoundingBox(
                extent=np.array([2.3, 1.0, 0.75], dtype=np.float32),
                location=np.array([0.0, 0.0, 0.75], dtype=np.float32),
            ),
            transform=Pose(
                translation=np.array([lead_x, lead_y, 0.0], dtype=np.float32),
                rotation=_yaw_rotation_matrix(lead_yaw),
            ),
            velocity=np.array(
                [8.0 * np.cos(lead_yaw), 8.0 * np.sin(lead_yaw), 0.0], dtype=np.float32
            ),
        )

        over_x, over_y, over_yaw = _overtaking_vehicle_state(
            t, rng_amplitude, rng_freq, total_duration
        )
        overtaker_actor = ActorState(
            id=overtaker_id,
            type_id="vehicle.audi.a2",
            bounding_box=BoundingBox(
                extent=np.array([2.0, 0.95, 0.7], dtype=np.float32),
                location=np.array([0.0, 0.0, 0.7], dtype=np.float32),
            ),
            transform=Pose(
                translation=np.array([over_x, over_y, 0.0], dtype=np.float32),
                rotation=_yaw_rotation_matrix(over_yaw),
            ),
            velocity=np.array(
                [11.0 * np.cos(over_yaw), 11.0 * np.sin(over_yaw), 0.0], dtype=np.float32
            ),
        )

        ped_x, ped_y, ped_yaw = _pedestrian_state(t, rng_amplitude, rng_freq)
        pedestrian_actor = ActorState(
            id=pedestrian_id,
            type_id="walker.pedestrian.0001",
            bounding_box=BoundingBox(
                extent=np.array([0.3, 0.3, 0.9], dtype=np.float32),
                location=np.array([0.0, 0.0, 0.9], dtype=np.float32),
            ),
            transform=Pose(
                translation=np.array([ped_x, ped_y, 0.0], dtype=np.float32),
                rotation=_yaw_rotation_matrix(ped_yaw),
            ),
            velocity=np.array(
                [1.2 * np.cos(ped_yaw), 1.2 * np.sin(ped_yaw), 0.0], dtype=np.float32
            ),
        )

        # Traffic light near x ~ 40m, cycling red/yellow/green every 60 ticks.
        cycle = frame_id % 60
        if cycle < 30:
            light_state = "red"
        elif cycle < 35:
            light_state = "yellow"
        else:
            light_state = "green"
        _, light_y_center, _ = _ego_position(40.0 / 8.0, rng_amplitude, rng_freq)
        traffic_light = TrafficLightState(
            id=traffic_light_id,
            state=light_state,
            position=np.array([40.0, light_y_center + 4.0, 3.0], dtype=np.float32),
        )

        # Lanes near the ego (a windowed slice keeps the log smaller and
        # matches what a real perception window would provide).
        window_mask = (x_grid > ego_x - 15.0) & (x_grid < ego_x + 60.0)
        lanes = [
            LaneSegment(lane_id=1, polyline=left_lane_polyline[window_mask]),
            LaneSegment(lane_id=2, polyline=right_lane_polyline[window_mask]),
        ]

        # Simple controller mock: proportional steer toward yaw, mild
        # throttle, brake only when the light is red and ego is close.
        steer = float(np.clip(-ego_yaw * 0.5, -1.0, 1.0))
        distance_to_light = 40.0 - ego_x
        braking = light_state == "red" and 0.0 < distance_to_light < 15.0
        control = ControlCommand(
            steer=steer,
            throttle=0.0 if braking else 0.45,
            brake=0.6 if braking else 0.0,
        )

        image_paths = {
            cam: [f"images/{cam}/{frame_id:06d}.png"] for cam in header.cameras
        }

        frame = TraceFrame(
            frame_id=frame_id,
            sim_time=round(t, 6),
            ego_pose_world=ego_pose,
            actors=[lead_actor, overtaker_actor, pedestrian_actor],
            lanes=lanes,
            traffic_lights=[traffic_light],
            plan=current_plan,
            control=control,
            image_paths=image_paths,
        )
        frames.append(frame)

    return header, frames


def write_trace(out_dir: Path, ticks: int, seed: int) -> Path:
    header, frames = build_frames(ticks, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header.to_json_dict(), sort_keys=True))
        f.write("\n")
        for frame in frames:
            f.write(json.dumps(frame.to_json_dict(), sort_keys=True))
            f.write("\n")
    return trace_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--ticks", type=int, default=200, help="number of ticks")
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    args = parser.parse_args(argv)

    trace_path = write_trace(args.out, args.ticks, args.seed)
    size_bytes = trace_path.stat().st_size
    with trace_path.open("r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    print(f"wrote {trace_path} ({n_lines} lines, {size_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
