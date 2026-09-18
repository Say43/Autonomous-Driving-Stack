"""Controlled CARLA regression scenarios on a dedicated server (no model / no network queue).

Two tests: stop before a stationary car; contain an intentionally sidewalk-bound plan.
Only actors spawned by this script are removed. Never use --port for another active run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import carla
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acarla.control import ControllerConfig, TrajectoryController  # noqa: E402
from acarla.control.supervisor import check_environment, footprint_is_drivable  # noqa: E402
from acarla.sim import groundtruth  # noqa: E402
from acarla.sim.client import SimSession  # noqa: E402
from acarla.types import BoundingBox, ControlCommand, PlanResult  # noqa: E402


def scenario(name, port, ticks):
    contacts = []
    obstacle = None
    with SimSession(port=port, map_name="Town10HD_Opt", tm_port=8100) as session:
        session.configure_sync(21, 21)
        ego = session.spawn_ego(
            random.Random(21), blueprint_filter="vehicle.lincoln.mkz_2017", autopilot=False
        )
        # Populate the synchronous actor snapshot before reading the spawn pose.
        session.tick()
        session.spawn_collision_sensor(ego, lambda event: contacts.append(event.frame))
        scene = groundtruth.SceneCache(session.world)
        bb = ego.bounding_box
        ego_box = BoundingBox(
            np.array([bb.extent.x, bb.extent.y, bb.extent.z], np.float32),
            np.array([bb.location.x, bb.location.y, bb.location.z], np.float32),
        )

        drivable = groundtruth.make_drivable_check(scene.map)

        controller = TrajectoryController(
            ControllerConfig(replan_interval_s=1, max_plan_age_s=1.25)
        )
        interventions = offroad = 0
        footprint_offroad = 0
        distance = 0.0
        last = None
        first_intervention = None
        try:
            if name == "stationary_car":
                spawn_pose = ego.get_transform()
                forward = spawn_pose.get_forward_vector()
                transform = carla.Transform(
                    carla.Location(
                        x=spawn_pose.location.x + 19 * forward.x,
                        y=spawn_pose.location.y + 19 * forward.y,
                        z=spawn_pose.location.z + 0.3,
                    ),
                    spawn_pose.rotation,
                )
                obstacle = session.world.try_spawn_actor(
                    session.world.get_blueprint_library().find("vehicle.lincoln.mkz_2020"),
                    transform,
                )
                if obstacle is None:
                    raise RuntimeError("could not spawn test obstacle")
                obstacle.apply_control(carla.VehicleControl(hand_brake=True))
                session.tick()
                actual = groundtruth.transform_to_pose(ego.get_transform())
                target = groundtruth.transform_to_pose(obstacle.get_transform())
                relative = actual.rotation.T @ (target.translation - actual.translation)
                if not (17 < relative[0] < 21 and abs(relative[1]) < 0.2):
                    raise RuntimeError(f"invalid test fixture: obstacle not ahead: {relative}")
            for tick in range(ticks):
                session.tick()
                p = groundtruth.transform_to_pose(ego.get_transform())
                velocity = ego.get_velocity()
                speed = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))
                if last is not None:
                    distance += float(np.linalg.norm(p.translation - last))
                last = p.translation.copy()
                if tick % 20 == 0:
                    x = np.arange(1, 65, dtype=np.float32) * 0.6
                    xyz = np.zeros((64, 3), np.float32)
                    xyz[:, 0] = x
                    if name == "sidewalk_plan":
                        xyz[:, 1] = -0.03 * x**2
                    plan = PlanResult(
                        tick,
                        xyz,
                        np.tile(np.eye(3, dtype=np.float32), (64, 1, 1)),
                        "SYNTHETIC SAFETY REGRESSION",
                        0,
                        "synthetic",
                    )
                    check = controller.set_plan(plan, p, tick * 0.05)
                    if not check.safe:
                        raise RuntimeError(f"test plan fails dynamics gate: {check.reason}")
                decision = controller.step(p, speed, tick * 0.05, 0.05)
                objects = groundtruth.actor_states(session.world, ego.id) + scene.nearby(
                    p.translation
                )
                guard = check_environment(
                    p,
                    speed,
                    decision.command,
                    controller.world_waypoints,
                    objects,
                    drivable,
                    ego_box,
                )
                command = decision.command
                if guard.brake:
                    interventions += 1
                    first_intervention = first_intervention or {
                        "tick": tick,
                        "speed_mps": speed,
                        "reason": guard.reason,
                        "distance_m": guard.hazard_distance_m,
                    }
                    command = ControlCommand(command.steer, 0, 1)
                    controller.reset_after_override()
                ego.apply_control(
                    carla.VehicleControl(
                        steer=command.steer, throttle=command.throttle, brake=command.brake
                    )
                )
                offroad += int(not drivable(p.translation))
                footprint_offroad += int(not footprint_is_drivable(p, ego_box, drivable))
                if contacts:
                    break
            time.sleep(0.2)  # allow the final asynchronous collision callback to arrive
            return {
                "scenario": name,
                "alpamayo_used": False,
                "ticks": tick + 1,
                "collisions": len(set(contacts)),
                "offroad_ticks": offroad,
                "footprint_offroad_ticks": footprint_offroad,
                "intervention_ticks": interventions,
                "first_intervention": first_intervention,
                "distance_m": distance,
                "final_speed_mps": speed,
                "passed": (
                    not contacts
                    and offroad == 0
                    and footprint_offroad == 0
                    and interventions > 0
                    and distance > 1
                ),
            }
        finally:
            if obstacle is not None and obstacle.is_alive:
                obstacle.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2100)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new output file")
    results = [
        scenario(name, args.port, args.ticks) for name in ("stationary_car", "sidewalk_plan")
    ]
    report = {"kind": "synthetic_safety_regression", "alpamayo_used": False, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(r["passed"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
