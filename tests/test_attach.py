"""Tests for acarla.sim.attach: rig-origin-to-actor-origin offset formula."""

from __future__ import annotations

import numpy as np
import pytest

from acarla.sim.attach import apply_offset, rig_to_actor_offset
from acarla.sim.rig import CameraSpec


def test_rig_to_actor_offset_example() -> None:
    offset = rig_to_actor_offset(
        rear_axle_to_bbox_center=1.354,
        bbox_extent_z=0.75,
        bbox_location_xyz=(0.0, 0.0, 0.75),
    )
    np.testing.assert_allclose(offset, [-1.354, 0.0, 0.0])


def test_rig_to_actor_offset_with_nonzero_bbox_xy() -> None:
    offset = rig_to_actor_offset(
        rear_axle_to_bbox_center=1.5,
        bbox_extent_z=0.8,
        bbox_location_xyz=(0.1, 0.2, 0.8),
    )
    np.testing.assert_allclose(offset, [-1.4, 0.2, 0.0])


def test_rig_to_actor_offset_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        rig_to_actor_offset(1.0, 0.5, (0.0, 0.0))


def test_apply_offset_translates_location_only() -> None:
    spec = CameraSpec(
        name="front_wide",
        fov=120.0,
        width=1920,
        height=1080,
        sensor_tick=0.1,
        x=1.6969041,
        y=0.010187792,
        z=1.40,
        roll=0.155,
        pitch=-0.443,
        yaw=0.489,
    )
    offset = np.array([-1.354, 0.0, 0.0])
    shifted = apply_offset(spec, offset)

    assert shifted.x == pytest.approx(spec.x - 1.354)
    assert shifted.y == pytest.approx(spec.y)
    assert shifted.z == pytest.approx(spec.z)
    assert shifted.roll == spec.roll
    assert shifted.pitch == spec.pitch
    assert shifted.yaw == spec.yaw
    assert shifted.name == spec.name
    assert shifted.fov == spec.fov


def test_apply_offset_rejects_bad_shape() -> None:
    spec = CameraSpec(
        name="front_wide",
        fov=120.0,
        width=1920,
        height=1080,
        sensor_tick=0.1,
        x=0.0,
        y=0.0,
        z=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    with pytest.raises(ValueError, match="shape"):
        apply_offset(spec, np.array([1.0, 2.0]))
