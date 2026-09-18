"""Public, synthetic reference-point regressions; no licensed calibration data."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from acarla.adapter.ego_frame import world_to_ego_history
from acarla.control.controller import ControllerConfig, TrajectoryController
from acarla.control.frames import offset_pose, plan_reference_points
from acarla.control.supervisor import SupervisorConfig, _path_samples
from acarla.types import ControlCommand, PlanResult, Pose, RunHeader, TraceFrame


def pose(x=0.0, y=0.0, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return Pose(
        np.array([x, y, 0], np.float32), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    )


def curve_plan(radius=30.0):
    """A right turn: physical rear axle on a circle, virtual rig 0.8 m ahead."""
    angles = np.arange(1, 65) * 0.3 / radius
    rear = np.column_stack((radius * np.sin(angles), radius * (1 - np.cos(angles)), np.zeros(64)))
    rotations = np.array([pose(yaw=a).rotation for a in angles])
    model_world = rear + rotations @ np.array([0.8, 0, 0])
    flip = np.diag([1, -1, 1])
    plan = PlanResult(
        0,
        ((model_world - [0.8, 0, 0]) @ flip).astype(np.float32),
        (flip @ rotations @ flip).astype(np.float32),
        None,
        0,
        "synthetic",
    )
    return plan, rear, rotations


def test_camera_rig_and_history_have_same_origin_through_turn():
    # Camera = actor + R*(model offset + camera-in-rig).
    offset = np.array([-0.6, 0.0, -0.03])
    camera = np.array([1.2, -0.4, 1.5])
    for yaw in (0, 0.3, math.pi / 2, -math.pi / 2):
        actor = pose(10, 20, yaw)
        model = offset_pose(actor, offset)
        np.testing.assert_allclose(
            model.translation + model.rotation @ camera,
            actor.translation + actor.rotation @ (offset + camera),
            atol=2e-6,
        )


def test_history_offset_must_rotate_at_each_sample():
    # Rotating at a fixed actor origin STILL moves an offset camera rig.
    from scipy.spatial.transform import Rotation

    from acarla.sim.coords import carla_rotation_to_rig_rotation, carla_to_rig_xyz

    actors = [pose(yaw=a) for a in np.linspace(0, 0.3, 16)]
    models = [offset_pose(p, (-0.6, 0, 0)) for p in actors]
    xyz = np.array([carla_to_rig_xyz(p.translation) for p in models], np.float64)
    quat = np.array(
        [Rotation.from_matrix(carla_rotation_to_rig_rotation(p.rotation)).as_quat() for p in models]
    )
    history, _ = world_to_ego_history(xyz, quat)
    assert np.linalg.norm(history[0, :2]) > 0.1
    np.testing.assert_allclose(history[-1], 0, atol=1e-6)


def test_predicted_rotation_transfers_model_path_to_rear_axle_and_actor():
    plan, rear, rotations = curve_plan()
    model = pose(0.8)
    np.testing.assert_allclose(plan_reference_points(plan, model, (-0.8, 0, 0)), rear, atol=3e-6)
    np.testing.assert_allclose(
        plan_reference_points(plan, model, (0.6, 0, 0)),
        rear + rotations @ np.array([1.4, 0, 0]),
        atol=3e-6,
    )


@pytest.mark.parametrize("yaw", [0, 0.8, -1.2])
def test_controller_tracks_rear_axle_not_camera_or_actor(yaw):
    plan, rear, rotations = curve_plan()
    actor = pose(12, -8, yaw)
    cfg = ControllerConfig(
        model_origin_in_actor=(-0.6, 0, 0),
        rear_axle_in_actor=(-1.4, 0, 0),
        max_steer_rate_per_s=100,
    )
    controller = TrajectoryController(cfg)
    assert controller.set_plan(plan, actor, 0).safe
    decision = controller.step(actor, 3, 0, 0.05)
    expected = math.atan(cfg.pure_pursuit.wheelbase_m / 30) / cfg.pure_pursuit.max_steer_angle_rad
    assert decision.command.steer == pytest.approx(expected, abs=2e-6)
    # The supervisor consumes ACTOR poses; the returned path must not be rear poses.
    expected_actor = rear + rotations @ np.array([1.4, 0, 0]) - [1.4, 0, 0]
    np.testing.assert_allclose(
        controller.world_waypoints, expected_actor @ actor.rotation.T + actor.translation, atol=5e-6
    )


def test_supervisor_steering_rollout_rotates_about_rear_axle():
    cfg = SupervisorConfig(rear_axle_in_actor=(-1.4, 0, 0), sample_m=1)
    command = ControlCommand(steer=0.2, throttle=0, brake=0)
    samples = list(_path_samples(pose(), command, None, 5, cfg))
    np.testing.assert_allclose(samples[0][1], [0, 0, 0], atol=1e-9)
    distance, point, yaw, _ = samples[-1]
    k = math.tan(command.steer * cfg.max_steer_angle_rad) / cfg.wheelbase_m
    rear = np.array([-1.4 + math.sin(yaw) / k, (1 - math.cos(yaw)) / k])
    expected_actor = rear + [1.4 * math.cos(yaw), 1.4 * math.sin(yaw)]
    np.testing.assert_allclose(point[:2], expected_actor, atol=1e-9)
    assert yaw == pytest.approx(distance * k)


def test_trace_distinguishes_actor_model_and_legacy_roundtrip():
    frame = TraceFrame(0, 0, pose(10), [], [], [], None, ControlCommand(0, 0, 0))
    assert frame.inference_pose_world is frame.ego_pose_world
    assert "model_pose_world" not in frame.to_json_dict()
    shifted = replace(frame, model_pose_world=pose(9.4))
    restored = TraceFrame.from_json_dict(shifted.to_json_dict())
    np.testing.assert_allclose(restored.ego_pose_world.translation, [10, 0, 0])
    np.testing.assert_allclose(restored.inference_pose_world.translation, [9.4, 0, 0])
    header = RunHeader(
        "test",
        "test",
        0,
        0,
        "test",
        "test",
        "test",
        "test",
        [],
        0.05,
        vehicle_geometry={"model_origin_in_actor": [-0.6, 0, 0]},
    )
    assert (
        RunHeader.from_json_dict(header.to_json_dict()).vehicle_geometry == header.vehicle_geometry
    )


@pytest.mark.parametrize("offset", [(float("nan"), 0, 0), (1, 2), (0, float("inf"), 0)])
def test_invalid_offsets_rejected(offset):
    with pytest.raises(ValueError, match="finite three-vector"):
        offset_pose(pose(), offset)


def test_supervisor_uses_body_heading_not_shifted_path_tangent():
    path = np.column_stack((np.arange(1, 10), np.arange(1, 10) * 0.2, np.zeros(9)))
    samples = list(
        _path_samples(pose(), ControlCommand(0, 0, 0), path, 4, SupervisorConfig(), np.zeros(9))
    )
    planned = [s for s in samples if s[3] == "plan"]
    assert planned
    assert all(s[2] == pytest.approx(0) for s in planned)


def test_offline_packet_uses_recorded_model_pose(tmp_path, monkeypatch):
    """Exercise the real packet-builder path without private rigs or image assets."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from acarla.record.writer import TraceWriter
    from acarla.sim.coords import carla_rotation_to_rig_rotation, carla_to_rig_xyz
    from acarla.types import ALPAMAYO_CAMERA_INDICES

    source = Path(__file__).parents[1] / "scripts/build_model_inputs.py"
    spec = importlib.util.spec_from_file_location("frame_test_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    cameras = list(ALPAMAYO_CAMERA_INDICES)
    specs = [SimpleNamespace(name=c, width=8, height=8, fov=90) for c in cameras]
    monkeypatch.setattr(builder, "load_rig_config", lambda _: specs)
    monkeypatch.setattr(builder, "load_rig_intrinsics", lambda _: dict.fromkeys(cameras))
    monkeypatch.setattr(builder, "build_remap", lambda *_: (None, None))
    monkeypatch.setattr(
        builder, "_load_and_remap_image", lambda *_: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    header = RunHeader("test", "test", 0, 0, "test", "test", "test", "test", cameras, 0.05)
    models = []
    with TraceWriter(tmp_path / "run", header) as writer:
        for i in range(33):
            actor = pose(yaw=i * 0.01)
            model = offset_pose(actor, (-0.6, 0, 0))
            models.append(model)
            writer.write_frame(
                TraceFrame(
                    i,
                    i * 0.05,
                    actor,
                    [],
                    [],
                    [],
                    None,
                    ControlCommand(0, 0, 0),
                    image_paths={c: [f"{i}.png"] for c in cameras} if i % 2 == 0 else {},
                    model_pose_world=model,
                )
            )
    arrays, _ = builder.build_model_inputs(tmp_path / "run", tmp_path / "unused", 1, 1)
    history = models[0:31:2]
    xyz = np.array([carla_to_rig_xyz(p.translation) for p in history], np.float64)
    quat = np.array(
        [
            Rotation.from_matrix(carla_rotation_to_rig_rotation(p.rotation)).as_quat()
            for p in history
        ]
    )
    expected, _ = world_to_ego_history(xyz, quat)
    actual = arrays["ego_history_xyz"][0, 0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.linalg.norm(actual[0]) > 0.1  # actor history alone would be identically zero
