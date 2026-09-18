#!/usr/bin/env python3
"""M5 controller baseline: close the CARLA loop without claiming Alpamayo.

The local 6 GiB GPU cannot run CARLA and Alpamayo together.  This executable
therefore replaces only the planner with deterministic CARLA lane waypoints;
the actual Pure Pursuit, PID, plan-age handling, safety gate, fail-safe braking,
actor lifecycle, and trace recorder are the same components intended for model
plans.  A successful run validates controller infrastructure, not M5's final
Alpamayo closed-loop requirement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import carla
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.control import TrajectoryController, world_points_to_flu  # noqa: E402
from acarla.record.writer import TraceWriter  # noqa: E402
from acarla.sim import groundtruth  # noqa: E402
from acarla.sim.client import SimSession  # noqa: E402
from acarla.types import (  # noqa: E402
    N_PLAN_WAYPOINTS,
    SAMPLE_DT,
    PlanResult,
    RunHeader,
    TraceFrame,
)

DEFAULT_MAP = "Town01"
GROUND_TRUTH_RADIUS_M = 60.0
LANE_MERGE_DISTANCE_M = 12.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--ticks",
        type=int,
        default=1200,
        help="world ticks; 1200 at 0.05 s is the 60-second baseline gate",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--target-speed", type=float, default=8.0, help="m/s")
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--purge-actors", action="store_true")
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _config_hash(target_speed_mps: float) -> str:
    payload = json.dumps(
        {
            "planner": "carla_waypoint_baseline",
            "target_speed_mps": target_speed_mps,
            "replan_hz": 2.0,
            "controller": "pure_pursuit_pid_safety_v2",
            "lane_merge_distance_m": LANE_MERGE_DISTANCE_M,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _choose_next_waypoint(current: object, candidates: list[object]) -> object:
    """Pick a deterministic continuation, preferring the current lane."""
    if not candidates:
        raise RuntimeError("CARLA route ended before the 64-waypoint horizon")
    return min(
        candidates,
        key=lambda wp: (
            wp.road_id != current.road_id,
            wp.lane_id != current.lane_id,
            wp.road_id,
            wp.lane_id,
            wp.s,
        ),
    )


def build_baseline_plan(
    carla_map: object,
    ego_transform: object,
    ego_pose_world: object,
    target_speed_mps: float,
    frame_id: int,
    model_config_hash: str,
) -> PlanResult:
    """Build a 64-point, 10 Hz FLU plan from CARLA lane centerlines."""
    if not math.isfinite(target_speed_mps) or target_speed_mps <= 0.0:
        raise ValueError("target_speed_mps must be finite and positive")
    spacing_m = target_speed_mps * SAMPLE_DT
    waypoint = carla_map.get_waypoint(ego_transform.location, project_to_road=True)
    if waypoint is None:
        raise RuntimeError("ego vehicle is not on a drivable CARLA waypoint")

    # ``Waypoint.next(distance)`` is not metrically exact around junctions:
    # Town01 returned individual 0.8 m requests as gaps up to about 0.99 m,
    # which looks like an impossible 17.5 m/s^2 acceleration to the safety
    # gate. Build a fine route polyline first, then sample it by measured arc
    # length. The controller therefore sees the requested 10 Hz speed profile,
    # independent of the map graph's waypoint spacing.
    ego_location = ego_transform.location
    dense_xyz = [[ego_location.x, ego_location.y, ego_location.z]]
    cumulative_m = [0.0]
    dense_step_m = min(0.25, spacing_m / 4.0)
    required_m = spacing_m * N_PLAN_WAYPOINTS + spacing_m
    for _ in range(N_PLAN_WAYPOINTS * 16):
        candidates = list(waypoint.next(dense_step_m))
        waypoint = _choose_next_waypoint(waypoint, candidates)
        loc = waypoint.transform.location
        point = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        previous = np.asarray(dense_xyz[-1], dtype=np.float64)
        segment_m = float(np.linalg.norm(point - previous))
        if segment_m <= 1e-6:
            continue
        dense_xyz.append(point.tolist())
        cumulative_m.append(cumulative_m[-1] + segment_m)
        if cumulative_m[-1] >= required_m:
            break
    if cumulative_m[-1] < required_m:
        raise RuntimeError(
            "CARLA route ended before the 64-waypoint horizon "
            f"({cumulative_m[-1]:.3f} m available, {required_m:.3f} m required)"
        )

    targets_m = spacing_m * np.arange(1, N_PLAN_WAYPOINTS + 1)
    dense_points = np.asarray(dense_xyz, dtype=np.float64)
    cumulative = np.asarray(cumulative_m, dtype=np.float64)
    world_points = np.column_stack(
        [np.interp(targets_m, cumulative, dense_points[:, axis]) for axis in range(3)]
    ).astype(np.float32)
    flu_points = world_points_to_flu(world_points, ego_pose_world)

    # Pure Pursuit can cut toward the inside of a bend, leaving the ego a
    # short distance from the lane centre at the next replan. A direct line
    # back to the centreline would make the first two samples appear to demand
    # excessive curvature. Blend from the ego's current forward tangent into
    # the map path over 12 m with a quintic smoothstep (zero slope at both
    # ends). The distance was checked against the tight Town01 bend that
    # exposed the failure, while leaving the audited 0.33 1/m curvature and
    # 9.8 m/s^2 acceleration limits untouched.
    blend_t = np.clip(targets_m / LANE_MERGE_DISTANCE_M, 0.0, 1.0)
    blend = 6.0 * blend_t**5 - 15.0 * blend_t**4 + 10.0 * blend_t**3
    straight_flu = np.column_stack(
        (targets_m, np.zeros(N_PLAN_WAYPOINTS), flu_points[:, 2])
    )
    flu_points = (
        straight_flu + (flu_points - straight_flu) * blend[:, None]
    ).astype(np.float32)

    origins = np.concatenate(
        (np.zeros((1, 2), dtype=np.float32), flu_points[:-1, :2]), axis=0
    )
    directions = flu_points[:, :2] - origins
    yaw = np.arctan2(directions[:, 1], directions[:, 0])
    rotations = np.zeros((N_PLAN_WAYPOINTS, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.cos(yaw)
    rotations[:, 0, 1] = -np.sin(yaw)
    rotations[:, 1, 0] = np.sin(yaw)
    rotations[:, 1, 1] = np.cos(yaw)
    rotations[:, 2, 2] = 1.0

    return PlanResult(
        frame_id=frame_id,
        waypoints_xyz=flu_points,
        waypoints_rot=rotations,
        reasoning="CARLA waypoint baseline; not Alpamayo output",
        inference_ms=0.0,
        model_config_hash=model_config_hash,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ticks <= 0:
        raise ValueError("--ticks must be positive")
    rng = random.Random(args.seed)
    config_hash = _config_hash(args.target_speed)
    collision_frames: list[int] = []
    fixed_delta_seconds = 0.05

    with SimSession(
        host=args.host,
        port=args.port,
        map_name=args.map,
        timeout_s=args.timeout,
        fixed_delta_seconds=fixed_delta_seconds,
    ) as session:
        session.configure_sync(args.seed, args.seed)
        if args.purge_actors:
            print(f"purged {session.purge_actors()} pre-existing actor(s)")
        ego = session.spawn_ego(rng, autopilot=False)
        session.spawn_traffic(args.traffic, rng, exclude_spawn_point=ego.get_transform())
        session.spawn_collision_sensor(
            ego, lambda event: collision_frames.append(int(event.frame))
        )

        header = RunHeader(
            carla_version=session.client.get_client_version(),
            map_name=session.world.get_map().name,
            seed_world=args.seed,
            seed_traffic_manager=args.seed,
            model_config_name="carla-waypoint-m5-controller-baseline",
            model_config_hash=config_hash,
            git_sha=_git_sha(),
            run_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            cameras=[],
            fixed_delta_seconds=session.fixed_delta_seconds,
        )
        controller = TrajectoryController()
        current_plan: PlanResult | None = None
        unsafe_replans = 0
        replans = 0
        ticks_completed = 0
        distance_m = 0.0
        speeds_mps: list[float] = []
        previous_xyz: np.ndarray | None = None
        offroad_ticks = 0
        max_lane_center_error_m = 0.0

        with TraceWriter(args.out, header) as writer:
            for local_frame in range(args.ticks):
                session.tick()
                sim_time = local_frame * session.fixed_delta_seconds
                ego_transform = ego.get_transform()
                ego_pose = groundtruth.transform_to_pose(ego_transform)

                if controller.needs_replan(sim_time):
                    current_plan = build_baseline_plan(
                        session.world.get_map(),
                        ego_transform,
                        ego_pose,
                        args.target_speed,
                        local_frame,
                        config_hash,
                    )
                    safety = controller.set_plan(current_plan, ego_pose, sim_time)
                    replans += 1
                    if not safety.safe:
                        unsafe_replans += 1

                velocity = ego.get_velocity()
                speed_mps = math.sqrt(
                    velocity.x * velocity.x
                    + velocity.y * velocity.y
                    + velocity.z * velocity.z
                )
                speeds_mps.append(speed_mps)
                decision = controller.step(
                    ego_pose,
                    current_speed_mps=speed_mps,
                    sim_time_s=sim_time,
                    dt=session.fixed_delta_seconds,
                )

                ego.apply_control(
                    carla.VehicleControl(
                        steer=decision.command.steer,
                        throttle=decision.command.throttle,
                        brake=decision.command.brake,
                    )
                )

                location = ego_transform.location
                location_xyz = np.array(
                    [location.x, location.y, location.z], dtype=np.float64
                )
                if previous_xyz is not None:
                    distance_m += float(np.linalg.norm(location_xyz - previous_xyz))
                previous_xyz = location_xyz
                lane_waypoint = session.world.get_map().get_waypoint(
                    location,
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving,
                )
                if lane_waypoint is None:
                    offroad_ticks += 1
                else:
                    max_lane_center_error_m = max(
                        max_lane_center_error_m,
                        location.distance(lane_waypoint.transform.location),
                    )
                writer.write_frame(
                    TraceFrame(
                        frame_id=local_frame,
                        sim_time=round(sim_time, 6),
                        ego_pose_world=ego_pose,
                        actors=groundtruth.actor_states(
                            session.world, ego.id, GROUND_TRUTH_RADIUS_M
                        ),
                        lanes=groundtruth.lane_segments(
                            session.world.get_map(), location, GROUND_TRUTH_RADIUS_M
                        ),
                        traffic_lights=groundtruth.traffic_light_states(
                            session.world, location, GROUND_TRUTH_RADIUS_M
                        ),
                        plan=current_plan,
                        control=decision.command,
                        image_paths={},
                    )
                )
                ticks_completed += 1
                if collision_frames:
                    break

    simulated_seconds = ticks_completed * fixed_delta_seconds
    operationally_valid = (
        ticks_completed == args.ticks
        and not collision_frames
        and unsafe_replans == 0
        and offroad_ticks == 0
    )
    passed = operationally_valid and simulated_seconds >= 60.0
    if passed:
        status = "controller_baseline_passed"
    elif operationally_valid:
        status = "controller_baseline_incomplete_duration"
    else:
        status = "controller_baseline_failed"
    summary = {
        "status": status,
        "alpamayo_used": False,
        "m5_passed": False,
        "ticks_requested": args.ticks,
        "ticks_completed": ticks_completed,
        "simulated_seconds": simulated_seconds,
        "distance_m": distance_m,
        "mean_speed_mps": sum(speeds_mps) / len(speeds_mps),
        "max_speed_mps": max(speeds_mps),
        "offroad_ticks": offroad_ticks,
        "max_lane_center_error_m": max_lane_center_error_m,
        "replans": replans,
        "unsafe_replans": unsafe_replans,
        "collision_frames": collision_frames,
        "model_config_hash": config_hash,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"trace: {args.out / 'trace.jsonl'}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
