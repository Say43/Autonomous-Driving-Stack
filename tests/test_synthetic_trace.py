"""Tests for scripts/make_synthetic_trace.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_synthetic_trace import REPLAN_INTERVAL, write_trace  # noqa: E402

from acarla.types import (  # noqa: E402
    ControlCommand,
    Pose,
    RunHeader,
    TraceFrame,
    assert_valid_rotation,
)


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return f.readlines()


def test_header_and_frames_deserialize(tmp_path: Path) -> None:
    trace_path = write_trace(tmp_path / "run", ticks=30, seed=0)
    lines = _read_lines(trace_path)

    assert len(lines) == 31  # 1 header + 30 frames

    header = RunHeader.from_json_dict(json.loads(lines[0]))
    assert header.map_name
    assert header.fixed_delta_seconds > 0

    frames = [TraceFrame.from_json_dict(json.loads(line)) for line in lines[1:]]
    assert len(frames) == 30
    for i, frame in enumerate(frames):
        assert frame.frame_id == i


def test_determinism_same_seed_identical_bytes(tmp_path: Path) -> None:
    path_a = write_trace(tmp_path / "run_a", ticks=25, seed=7)
    path_b = write_trace(tmp_path / "run_b", ticks=25, seed=7)

    assert path_a.read_bytes() == path_b.read_bytes()


def test_different_seed_differs(tmp_path: Path) -> None:
    path_a = write_trace(tmp_path / "run_a", ticks=25, seed=1)
    path_b = write_trace(tmp_path / "run_b", ticks=25, seed=2)

    assert path_a.read_bytes() != path_b.read_bytes()


def test_replanning_cadence(tmp_path: Path) -> None:
    trace_path = write_trace(tmp_path / "run", ticks=40, seed=3)
    lines = _read_lines(trace_path)
    frames = [TraceFrame.from_json_dict(json.loads(line)) for line in lines[1:]]

    for start in range(0, len(frames) - REPLAN_INTERVAL, REPLAN_INTERVAL):
        base_plan = frames[start].plan
        assert base_plan is not None
        for offset in range(1, REPLAN_INTERVAL):
            other_plan = frames[start + offset].plan
            assert other_plan is not None
            assert np.array_equal(base_plan.waypoints_xyz, other_plan.waypoints_xyz)
            assert np.array_equal(base_plan.waypoints_rot, other_plan.waypoints_rot)

    # At the replan boundary, the plan changes (the ego has moved on).
    changed = False
    for start in range(0, len(frames) - REPLAN_INTERVAL, REPLAN_INTERVAL):
        before = frames[start].plan
        after = frames[start + REPLAN_INTERVAL].plan
        if not np.array_equal(before.waypoints_xyz, after.waypoints_xyz):
            changed = True
    assert changed


def test_reasoning_only_every_third_replan(tmp_path: Path) -> None:
    trace_path = write_trace(tmp_path / "run", ticks=60, seed=4)
    lines = _read_lines(trace_path)
    frames = [TraceFrame.from_json_dict(json.loads(line)) for line in lines[1:]]

    replan_frames = [frames[i] for i in range(0, len(frames), REPLAN_INTERVAL)]
    reasonings = [f.plan.reasoning for f in replan_frames]

    assert any(r is not None for r in reasonings)
    assert any(r is None for r in reasonings)
    # Every third replan (0-indexed) should carry reasoning, the others not.
    for i, r in enumerate(reasonings):
        if i % 3 == 0:
            assert r is not None
        else:
            assert r is None


def test_all_rotation_matrices_valid(tmp_path: Path) -> None:
    trace_path = write_trace(tmp_path / "run", ticks=20, seed=5)
    lines = _read_lines(trace_path)
    frames = [TraceFrame.from_json_dict(json.loads(line)) for line in lines[1:]]

    for frame in frames:
        assert_valid_rotation(
            frame.ego_pose_world.rotation.reshape(1, 3, 3), "ego_pose_world.rotation"
        )
        for actor in frame.actors:
            assert_valid_rotation(
                actor.transform.rotation.reshape(1, 3, 3), "actor.transform.rotation"
            )
        if frame.plan is not None:
            assert_valid_rotation(frame.plan.waypoints_rot, "plan.waypoints_rot")


def test_frame_with_none_plan_is_serializable() -> None:
    frame = TraceFrame(
        frame_id=0,
        sim_time=0.0,
        ego_pose_world=Pose(
            translation=np.zeros(3, dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        ),
        actors=[],
        lanes=[],
        traffic_lights=[],
        plan=None,
        control=ControlCommand(steer=0.0, throttle=0.0, brake=0.0),
    )

    d = frame.to_json_dict()
    assert d["plan"] is None
    round_tripped = TraceFrame.from_json_dict(d)
    assert round_tripped.plan is None
