"""Tests for acarla.model.inputs."""

from __future__ import annotations

import numpy as np

from acarla.adapter.packet import build_sensor_packet
from acarla.model.inputs import sensor_packet_to_model_inputs
from acarla.types import N_EGO_WAYPOINTS, SAMPLE_DT

ALL_CAMS = ("cross_left", "front_wide", "cross_right", "front_tele")


def _make_frames(cam: str, h: int = 4, w: int = 6, fill: int = 0) -> np.ndarray:
    frames = np.zeros((4, h, w, 3), dtype=np.uint8)
    frames[..., :] = fill
    return frames


def _make_packet(cams: tuple[str, ...] = ALL_CAMS) -> object:
    frames_hwc = {cam: _make_frames(cam, fill=idx) for idx, cam in enumerate(cams)}
    image_timestamps = {cam: [0.0, 0.1, 0.2, 0.3] for cam in cams}
    n = N_EGO_WAYPOINTS
    world_xyz = np.zeros((n, 3), dtype=np.float64)
    world_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    ego_timestamps = [round(i * SAMPLE_DT, 4) for i in range(n)]

    return build_sensor_packet(
        frame_id=0,
        sim_time=1.0,
        frames_hwc=frames_hwc,
        image_timestamps=image_timestamps,
        world_xyz=world_xyz,
        world_quat_xyzw=world_quat_xyzw,
        ego_timestamps=ego_timestamps,
    )


def test_camera_order_independent_of_dict_insertion_order() -> None:
    shuffled = ("front_tele", "cross_left", "front_wide", "cross_right")
    packet = _make_packet(shuffled)

    inputs = sensor_packet_to_model_inputs(packet)

    np.testing.assert_array_equal(inputs["camera_indices"], np.array([0, 1, 2, 6]))
    assert inputs["camera_prompt_names"] == [
        "Front left camera",
        "Front camera",
        "Front right camera",
        "Front telephoto camera",
    ]


def test_chw_transpose_preserves_known_pixel() -> None:
    cams = ALL_CAMS
    frames_hwc = {cam: _make_frames(cam, h=4, w=6) for cam in cams}
    # Plant a known, distinct pixel in front_wide's first frame.
    frames_hwc["front_wide"][0, 2, 3] = [10, 20, 30]
    image_timestamps = {cam: [0.0, 0.1, 0.2, 0.3] for cam in cams}
    n = N_EGO_WAYPOINTS
    world_xyz = np.zeros((n, 3), dtype=np.float64)
    world_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    ego_timestamps = [round(i * SAMPLE_DT, 4) for i in range(n)]

    packet = build_sensor_packet(
        frame_id=0,
        sim_time=1.0,
        frames_hwc=frames_hwc,
        image_timestamps=image_timestamps,
        world_xyz=world_xyz,
        world_quat_xyzw=world_quat_xyzw,
        ego_timestamps=ego_timestamps,
    )

    inputs = sensor_packet_to_model_inputs(packet)

    # front_wide has Alpamayo index 1 -> position 1 in the sorted [0,1,2,6] order.
    front_wide_frames = inputs["image_frames"][1]
    assert front_wide_frames.shape == (4, 3, 4, 6)
    np.testing.assert_array_equal(front_wide_frames[0, :, 2, 3], [10, 20, 30])


def test_shapes_and_dtypes() -> None:
    packet = _make_packet()

    inputs = sensor_packet_to_model_inputs(packet)

    assert inputs["image_frames"].shape == (4, 4, 3, 4, 6)
    assert inputs["image_frames"].dtype == np.uint8
    assert inputs["camera_indices"].shape == (4,)
    assert inputs["camera_indices"].dtype == np.int64
    assert inputs["ego_history_xyz"].shape == (1, 1, N_EGO_WAYPOINTS, 3)
    assert inputs["ego_history_xyz"].dtype == np.float32
    assert inputs["ego_history_rot"].shape == (1, 1, N_EGO_WAYPOINTS, 3, 3)
    assert inputs["ego_history_rot"].dtype == np.float32
    assert len(inputs["camera_prompt_names"]) == 4


def test_two_camera_subset_works() -> None:
    packet = _make_packet(("front_wide", "front_tele"))

    inputs = sensor_packet_to_model_inputs(packet)

    np.testing.assert_array_equal(inputs["camera_indices"], np.array([1, 6]))
    assert inputs["image_frames"].shape == (2, 4, 3, 4, 6)
    assert inputs["camera_prompt_names"] == ["Front camera", "Front telephoto camera"]
