"""Read-only off-policy audit of recorded poses, NOT a new closed-loop success.

Uses a dedicated live CARLA map for lane/mesh queries. Never ticks the world,
spawns actors, changes settings, or writes to the input trace/results directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import carla
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acarla.control.frames import flu_points_to_world  # noqa: E402
from acarla.control.supervisor import check_environment  # noqa: E402
from acarla.record.reader import read_trace  # noqa: E402
from acarla.sim.groundtruth import SceneCache, make_drivable_check  # noqa: E402
from acarla.types import BoundingBox  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2100)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new output file")
    if args.stride < 1:
        raise ValueError("stride must be positive")
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30)
    world = client.get_world()
    header, iterator = read_trace(args.trace)
    if world.get_map().name.rsplit("/", 1)[-1] != header.map_name.rsplit("/", 1)[-1]:
        raise ValueError("CARLA map does not match the trace")
    scene = SceneCache(world)
    frames = list(iterator)
    by_id = {f.frame_id: f for f in frames}
    ego_box = BoundingBox(
        np.array([2.45, 1.05, 0.8], np.float32), np.array([0, 0, 0.8], np.float32)
    )

    drivable = make_drivable_check(scene.map)

    decisions = []
    for i in range(1, len(frames), args.stride):
        f, previous = frames[i], frames[i - 1]
        if f.plan is None:
            continue
        dt = f.sim_time - previous.sim_time
        speed = float(
            np.linalg.norm(f.ego_pose_world.translation - previous.ego_pose_world.translation) / dt
        )
        source = by_id.get(f.plan.frame_id)
        if source is None:
            continue
        path = flu_points_to_world(f.plan.waypoints_xyz, source.ego_pose_world)
        static = scene.nearby(f.ego_pose_world.translation)
        decision = check_environment(
            f.ego_pose_world, speed, f.control, path, f.actors + static, drivable, ego_box
        )
        decisions.append(
            {"frame_id": f.frame_id, "sim_time": f.sim_time, "speed_mps": speed, **asdict(decision)}
        )
    counts = Counter(d["reason"] for d in decisions if d["brake"])
    result = {
        "kind": "off_policy_replay_audit",
        "closed_loop_pass": False,
        "limitation": (
            "Interventions on recorded states do not prove the subsequent closed loop safe."
        ),
        "source_trace": str(args.trace.resolve()),
        "map": header.map_name,
        "static_objects": len(scene.objects),
        "checked_frames": len(decisions),
        "brake_frames": sum(d["brake"] for d in decisions),
        "reasons": dict(counts),
        "decisions": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
