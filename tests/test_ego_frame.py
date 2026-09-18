"""Tests for acarla.adapter.ego_frame."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from acarla.adapter.ego_frame import (
    carla_pose_to_world_quat,
    ego_to_world,
    world_to_ego_history,
)
from acarla.types import assert_valid_rotation

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def test_all_poses_equal_t0_gives_zero_translation_and_identity_rotation() -> None:
    n = 5
    world_xyz = np.tile(np.array([1.0, 2.0, 3.0]), (n, 1))
    world_quat = np.tile(IDENTITY_QUAT, (n, 1))

    xyz_local, rot_local = world_to_ego_history(world_xyz, world_quat)

    np.testing.assert_allclose(xyz_local, np.zeros((n, 3), dtype=np.float32), atol=1e-7)
    identity = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    np.testing.assert_allclose(rot_local, identity, atol=1e-6)


def test_yaw_90_at_t0_rotates_forward_point_correctly() -> None:
    # t0 (last entry) is yawed +90 degrees about z. A point 1 m ahead of the
    # vehicle in world coordinates (along +x, since t0 faces +y after the
    # yaw) should land at local (0, 1, 0) once frozen into the t0 ego frame.
    t0_quat = Rotation.from_euler("z", 90, degrees=True).as_quat()
    world_quat = np.stack([IDENTITY_QUAT, t0_quat])
    t0_world_xyz = np.array([0.0, 0.0, 0.0])
    point_world_xyz = np.array([1.0, 0.0, 0.0])
    world_xyz = np.stack([point_world_xyz, t0_world_xyz])

    xyz_local, rot_local = world_to_ego_history(world_xyz, world_quat)

    # xyz_local[-1] is t0 itself -> zero.
    np.testing.assert_allclose(xyz_local[-1], np.zeros(3), atol=1e-6)
    # xyz_local[0] is the point, expressed in the t0 (yawed) ego frame.
    expected = Rotation.from_quat(t0_quat).inv().apply(point_world_xyz - t0_world_xyz)
    np.testing.assert_allclose(xyz_local[0], expected.astype(np.float32), atol=1e-6)


def test_roundtrip_world_to_ego_to_world() -> None:
    rng = np.random.default_rng(0)
    n = 16
    world_xyz = rng.normal(size=(n, 3)) * 10.0
    random_rotvecs = rng.normal(size=(n, 3))
    world_quat = Rotation.from_rotvec(random_rotvecs).as_quat()

    xyz_local, rot_local = world_to_ego_history(world_xyz, world_quat)

    t0_xyz = world_xyz[-1]
    t0_quat = world_quat[-1]
    world_xyz_rt, world_quat_rt = ego_to_world(
        xyz_local.astype(np.float64), rot_local.astype(np.float64), t0_xyz, t0_quat
    )

    np.testing.assert_allclose(world_xyz_rt, world_xyz, atol=1e-6)

    # Compare rotations via relative rotation angle, robust to quaternion sign.
    rel = Rotation.from_quat(world_quat_rt) * Rotation.from_quat(world_quat).inv()
    angles = rel.magnitude()
    assert np.all(angles < 1e-6)


def test_output_dtypes_and_valid_rotations() -> None:
    rng = np.random.default_rng(1)
    n = 16
    world_xyz = rng.normal(size=(n, 3))
    world_quat = Rotation.from_rotvec(rng.normal(size=(n, 3))).as_quat()

    xyz_local, rot_local = world_to_ego_history(world_xyz, world_quat)

    assert xyz_local.dtype == np.float32
    assert rot_local.dtype == np.float32
    assert_valid_rotation(rot_local, "rot_local")


def test_carla_pose_to_world_quat_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        carla_pose_to_world_quat()
