"""Tests for acarla.types."""

from __future__ import annotations

import numpy as np
import pytest

from acarla.types import (
    ALPAMAYO_CAMERA_INDICES,
    N_EGO_WAYPOINTS,
    N_HISTORY_FRAMES,
    N_PLAN_WAYPOINTS,
    SAMPLE_DT,
    ActorState,
    BoundingBox,
    ControlCommand,
    LaneSegment,
    PlanResult,
    Pose,
    RunHeader,
    SensorPacket,
    TraceFrame,
    TrafficLightState,
    alpamayo_camera_order,
    assert_valid_rotation,
)

IDENTITY = np.eye(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory helpers -- keep tests readable
# ---------------------------------------------------------------------------


def make_rotations(n: int) -> np.ndarray:
    """n copies of the identity rotation, shape (n, 3, 3) float32."""
    return np.tile(IDENTITY, (n, 1, 1)).astype(np.float32)


def make_sensor_packet(
    cameras: tuple[str, ...] = ("front_wide", "cross_left")
) -> SensorPacket:
    images = {
        cam: np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8) for cam in cameras
    }
    image_timestamps = {
        cam: [i * SAMPLE_DT for i in range(N_HISTORY_FRAMES)] for cam in cameras
    }
    return SensorPacket(
        frame_id=0,
        sim_time=0.0,
        images=images,
        image_timestamps=image_timestamps,
        ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
        ego_rotation=make_rotations(N_EGO_WAYPOINTS),
        ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
        command="follow_lane",
        nav_guidance=None,
    )


def make_plan_result() -> PlanResult:
    return PlanResult(
        frame_id=0,
        waypoints_xyz=np.zeros((N_PLAN_WAYPOINTS, 3), dtype=np.float32),
        waypoints_rot=make_rotations(N_PLAN_WAYPOINTS),
        reasoning="go straight",
        inference_ms=12.5,
        model_config_hash="abc123",
    )


def make_pose() -> Pose:
    return Pose(
        translation=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        rotation=IDENTITY.copy(),
    )


def make_actor() -> ActorState:
    return ActorState(
        id=7,
        type_id="vehicle.tesla.model3",
        bounding_box=BoundingBox(
            extent=np.array([2.0, 1.0, 0.75], dtype=np.float32),
            location=np.zeros(3, dtype=np.float32),
        ),
        transform=make_pose(),
        velocity=np.array([3.0, 0.0, 0.0], dtype=np.float32),
    )


def make_lane() -> LaneSegment:
    return LaneSegment(
        lane_id=42,
        polyline=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
    )


def make_traffic_light() -> TrafficLightState:
    return TrafficLightState(
        id=3, state="red", position=np.array([5.0, 5.0, 3.0], dtype=np.float32)
    )


def make_control() -> ControlCommand:
    return ControlCommand(steer=0.1, throttle=0.5, brake=0.0)


def make_trace_frame(with_plan: bool = True) -> TraceFrame:
    return TraceFrame(
        frame_id=0,
        sim_time=0.0,
        ego_pose_world=make_pose(),
        actors=[make_actor()],
        lanes=[make_lane()],
        traffic_lights=[make_traffic_light()],
        plan=make_plan_result() if with_plan else None,
        control=make_control(),
        image_paths={"front": ["frame_000000_front.png"]},
    )


def make_run_header() -> RunHeader:
    return RunHeader(
        carla_version="0.9.16",
        map_name="Town01",
        seed_world=1234,
        seed_traffic_manager=5678,
        model_config_name="alpamayo-1.5-vla-base",
        model_config_hash="deadbeef",
        git_sha="0123456789abcdef",
        run_timestamp="2026-09-04T12:00:00Z",
        cameras=["front", "left", "right"],
        fixed_delta_seconds=0.05,
    )


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------


def test_sensor_packet_valid():
    packet = make_sensor_packet()
    assert packet.frame_id == 0
    assert packet.images["front_wide"].shape == (N_HISTORY_FRAMES, 8, 8, 3)


def test_sensor_packet_two_cameras_valid():
    packet = make_sensor_packet(cameras=("front_wide", "front_tele"))
    assert set(packet.images.keys()) == {"front_wide", "front_tele"}


def test_plan_result_valid():
    plan = make_plan_result()
    assert plan.waypoints_xyz.shape == (N_PLAN_WAYPOINTS, 3)


def test_pose_valid():
    pose = make_pose()
    assert pose.translation.shape == (3,)


def test_actor_state_valid():
    actor = make_actor()
    assert actor.type_id == "vehicle.tesla.model3"


def test_lane_segment_valid():
    lane = make_lane()
    assert lane.polyline.shape[0] == 3


def test_traffic_light_valid():
    tl = make_traffic_light()
    assert tl.state == "red"


def test_control_command_valid():
    c = make_control()
    assert c.brake == 0.0


def test_trace_frame_valid():
    frame = make_trace_frame()
    assert frame.plan is not None
    assert frame.image_paths["front"] == ["frame_000000_front.png"]


def test_run_header_valid():
    header = make_run_header()
    assert header.map_name == "Town01"


def test_assert_valid_rotation_accepts_identity():
    assert_valid_rotation(IDENTITY, "rot")
    assert_valid_rotation(make_rotations(5), "rot")


# ---------------------------------------------------------------------------
# Negative tests: wrong shape
# ---------------------------------------------------------------------------


def test_sensor_packet_wrong_image_shape_raises():
    with pytest.raises(ValueError, match="images"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8), dtype=np.uint8)}
        image_timestamps = {"front_wide": [0.0, 0.1, 0.2, 0.3]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


def test_plan_result_wrong_waypoint_count_raises():
    with pytest.raises(ValueError, match="waypoints_xyz"):
        PlanResult(
            frame_id=0,
            waypoints_xyz=np.zeros((10, 3), dtype=np.float32),
            waypoints_rot=make_rotations(N_PLAN_WAYPOINTS),
            reasoning=None,
            inference_ms=1.0,
            model_config_hash="x",
        )


def test_lane_segment_too_short_raises():
    with pytest.raises(ValueError, match="polyline"):
        LaneSegment(lane_id=1, polyline=np.zeros((1, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Negative tests: wrong dtype
# ---------------------------------------------------------------------------


def test_sensor_packet_wrong_image_dtype_raises():
    with pytest.raises(ValueError, match="dtype"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.float32)}
        image_timestamps = {"front_wide": [0.0, 0.1, 0.2, 0.3]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


def test_plan_result_wrong_dtype_raises():
    with pytest.raises(ValueError, match="dtype"):
        PlanResult(
            frame_id=0,
            waypoints_xyz=np.zeros((N_PLAN_WAYPOINTS, 3), dtype=np.float64),
            waypoints_rot=make_rotations(N_PLAN_WAYPOINTS),
            reasoning=None,
            inference_ms=1.0,
            model_config_hash="x",
        )


# ---------------------------------------------------------------------------
# Negative tests: non-monotonic timestamps
# ---------------------------------------------------------------------------


def test_sensor_packet_non_monotonic_image_timestamps_raises():
    with pytest.raises(ValueError, match="increasing"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
        image_timestamps = {"front_wide": [0.0, 0.2, 0.1, 0.3]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


def test_sensor_packet_non_monotonic_ego_timestamps_raises():
    with pytest.raises(ValueError, match="increasing"):
        packet_kwargs = dict(
            frame_id=0,
            sim_time=0.0,
            images={
                "front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)
            },
            image_timestamps={"front_wide": [i * SAMPLE_DT for i in range(N_HISTORY_FRAMES)]},
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            command="follow_lane",
            nav_guidance=None,
        )
        bad_ts = [i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)]
        bad_ts[3], bad_ts[4] = bad_ts[4], bad_ts[3]
        SensorPacket(ego_timestamps=bad_ts, **packet_kwargs)


def test_sensor_packet_non_strict_duplicate_timestamps_raises():
    with pytest.raises(ValueError, match="increasing"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
        image_timestamps = {"front_wide": [0.0, 0.1, 0.1, 0.2]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


# ---------------------------------------------------------------------------
# Negative tests: non-orthonormal rotation matrix
# ---------------------------------------------------------------------------


def test_assert_valid_rotation_rejects_non_orthonormal():
    bad = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="orthonormal"):
        assert_valid_rotation(bad, "rot")


def test_assert_valid_rotation_rejects_reflection():
    # Orthonormal but determinant -1 (a reflection, not a rotation).
    bad = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="determinant"):
        assert_valid_rotation(bad, "rot")


def test_plan_result_non_orthonormal_rotation_raises():
    bad_rot = make_rotations(N_PLAN_WAYPOINTS)
    bad_rot[0, 0, 0] = 5.0
    with pytest.raises(ValueError, match="orthonormal"):
        PlanResult(
            frame_id=0,
            waypoints_xyz=np.zeros((N_PLAN_WAYPOINTS, 3), dtype=np.float32),
            waypoints_rot=bad_rot,
            reasoning=None,
            inference_ms=1.0,
            model_config_hash="x",
        )


# ---------------------------------------------------------------------------
# Negative tests: inconsistent camera keys
# ---------------------------------------------------------------------------


def test_sensor_packet_inconsistent_camera_keys_raises():
    with pytest.raises(ValueError, match="camera keys"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
        image_timestamps = {"cross_left": [0.0, 0.1, 0.2, 0.3]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


# ---------------------------------------------------------------------------
# Negative tests: unknown camera name
# ---------------------------------------------------------------------------


def test_sensor_packet_unknown_camera_name_raises():
    with pytest.raises(ValueError, match="camera"):
        images = {"front": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
        image_timestamps = {"front": [i * SAMPLE_DT for i in range(N_HISTORY_FRAMES)]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


# ---------------------------------------------------------------------------
# Negative/positive tests: 10 Hz sample-rate contract
# ---------------------------------------------------------------------------


def test_sensor_packet_wrong_dt_image_timestamps_raises():
    with pytest.raises(ValueError, match="non-uniform sampling"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
        # Uniformly spaced, but on a 0.2 s grid instead of the required 0.1 s.
        image_timestamps = {"front_wide": [0.0, 0.2, 0.4, 0.6]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


def test_sensor_packet_jitter_beyond_tolerance_raises():
    with pytest.raises(ValueError, match="non-uniform sampling"):
        images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
        image_timestamps = {"front_wide": [0.0, 0.1, 0.2, 0.2015]}
        SensorPacket(
            frame_id=0,
            sim_time=0.0,
            images=images,
            image_timestamps=image_timestamps,
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
            command="follow_lane",
            nav_guidance=None,
        )


def test_sensor_packet_jitter_within_tolerance_accepted():
    images = {"front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)}
    image_timestamps = {"front_wide": [0.0, 0.1, 0.2, 0.3005]}
    packet = SensorPacket(
        frame_id=0,
        sim_time=0.0,
        images=images,
        image_timestamps=image_timestamps,
        ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
        ego_rotation=make_rotations(N_EGO_WAYPOINTS),
        ego_timestamps=[i * SAMPLE_DT for i in range(N_EGO_WAYPOINTS)],
        command="follow_lane",
        nav_guidance=None,
    )
    assert packet.image_timestamps["front_wide"][-1] == pytest.approx(0.3005)


def test_sensor_packet_wrong_dt_ego_timestamps_raises():
    with pytest.raises(ValueError, match="non-uniform sampling"):
        packet_kwargs = dict(
            frame_id=0,
            sim_time=0.0,
            images={
                "front_wide": np.zeros((N_HISTORY_FRAMES, 8, 8, 3), dtype=np.uint8)
            },
            image_timestamps={
                "front_wide": [i * SAMPLE_DT for i in range(N_HISTORY_FRAMES)]
            },
            ego_translation=np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float32),
            ego_rotation=make_rotations(N_EGO_WAYPOINTS),
            command="follow_lane",
            nav_guidance=None,
        )
        # Uniformly increasing but spaced 0.2 s apart instead of 0.1 s.
        bad_ts = [i * 0.2 for i in range(N_EGO_WAYPOINTS)]
        SensorPacket(ego_timestamps=bad_ts, **packet_kwargs)


# ---------------------------------------------------------------------------
# alpamayo_camera_order
# ---------------------------------------------------------------------------


def test_alpamayo_camera_order_default_four_cameras():
    shuffled = ["front_tele", "cross_right", "front_wide", "cross_left"]
    assert alpamayo_camera_order(shuffled) == [
        "cross_left",
        "front_wide",
        "cross_right",
        "front_tele",
    ]
    ordered_indices = [ALPAMAYO_CAMERA_INDICES[n] for n in alpamayo_camera_order(shuffled)]
    assert ordered_indices == [0, 1, 2, 6]


def test_alpamayo_camera_order_subset():
    assert alpamayo_camera_order(["front_tele", "front_wide"]) == [
        "front_wide",
        "front_tele",
    ]


def test_alpamayo_camera_order_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown camera"):
        alpamayo_camera_order(["front_wide", "back"])


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


def test_plan_result_json_roundtrip():
    plan = make_plan_result()
    restored = PlanResult.from_json_dict(plan.to_json_dict())
    np.testing.assert_array_equal(restored.waypoints_xyz, plan.waypoints_xyz)
    np.testing.assert_array_equal(restored.waypoints_rot, plan.waypoints_rot)
    assert restored.waypoints_xyz.dtype == plan.waypoints_xyz.dtype
    assert restored.frame_id == plan.frame_id
    assert restored.reasoning == plan.reasoning
    assert restored.inference_ms == plan.inference_ms
    assert restored.model_config_hash == plan.model_config_hash


def test_trace_frame_json_roundtrip():
    frame = make_trace_frame()
    restored = TraceFrame.from_json_dict(frame.to_json_dict())
    np.testing.assert_array_equal(
        restored.ego_pose_world.translation, frame.ego_pose_world.translation
    )
    np.testing.assert_array_equal(
        restored.ego_pose_world.rotation, frame.ego_pose_world.rotation
    )
    assert len(restored.actors) == len(frame.actors)
    np.testing.assert_array_equal(
        restored.actors[0].velocity, frame.actors[0].velocity
    )
    np.testing.assert_array_equal(
        restored.lanes[0].polyline, frame.lanes[0].polyline
    )
    assert restored.traffic_lights[0].state == frame.traffic_lights[0].state
    assert restored.plan is not None
    np.testing.assert_array_equal(
        restored.plan.waypoints_xyz, frame.plan.waypoints_xyz
    )
    assert restored.control == frame.control
    assert restored.image_paths == frame.image_paths


def test_trace_frame_json_roundtrip_without_plan():
    frame = make_trace_frame(with_plan=False)
    restored = TraceFrame.from_json_dict(frame.to_json_dict())
    assert restored.plan is None


def test_run_header_json_roundtrip():
    header = make_run_header()
    restored = RunHeader.from_json_dict(header.to_json_dict())
    assert restored == header
