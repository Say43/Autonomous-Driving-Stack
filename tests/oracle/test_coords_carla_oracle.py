"""Oracle tests for `src/acarla/sim/coords.py` against the real CARLA client.

Runs ONLY in `.venv-sim` (`pytest.importorskip("carla")` skips it cleanly
in the 3.13 env). `carla.Transform` is pure client-side math -- no running
CARLA server is required.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

carla = pytest.importorskip("carla")

from acarla.sim import coords  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "m2_reference"

ALPAMAYO_CAMERAS = {
    "cross_left": "camera_cross_left_120fov",
    "front_wide": "camera_front_wide_120fov",
    "cross_right": "camera_cross_right_120fov",
    "front_tele": "camera_front_tele_30fov",
}


def _load_extrinsics_row(sensor_name: str) -> dict:
    with open(GOLDEN_DIR / "calibration_sensor_extrinsics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["sensor_name"] == sensor_name:
                return {k: (float(v) if k != "sensor_name" else v) for k, v in row.items()}
    raise KeyError(sensor_name)


def _quat_xyzw_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def test_carla_euler_to_matrix_matches_get_matrix():
    rng = np.random.default_rng(42)
    for _ in range(200):
        pitch = rng.uniform(-180, 180)
        yaw = rng.uniform(-180, 180)
        roll = rng.uniform(-180, 180)
        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        t = carla.Transform(carla.Location(0, 0, 0), rotation)
        expected = np.array(t.get_matrix())[:3, :3]
        actual = coords.carla_euler_to_matrix(pitch, yaw, roll)
        assert np.allclose(actual, expected, atol=1e-6), (pitch, yaw, roll)


def test_get_matrix_is_orthonormal_det_plus_one():
    rng = np.random.default_rng(7)
    for _ in range(50):
        pitch = rng.uniform(-180, 180)
        yaw = rng.uniform(-180, 180)
        roll = rng.uniform(-180, 180)
        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        t = carla.Transform(carla.Location(0, 0, 0), rotation)
        m = np.array(t.get_matrix())[:3, :3]
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(m), 1.0, atol=1e-6)


def test_carla_euler_matrix_to_euler_roundtrip_matches_carla():
    rng = np.random.default_rng(99)
    for _ in range(200):
        pitch = rng.uniform(-89, 89)
        yaw = rng.uniform(-180, 180)
        roll = rng.uniform(-89, 89)
        r = coords.carla_euler_to_matrix(pitch, yaw, roll)
        p2, y2, r2 = coords.matrix_to_carla_euler(r)
        t = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(pitch=p2, yaw=y2, roll=r2))
        expected = np.array(t.get_matrix())[:3, :3]
        assert np.allclose(r, expected, atol=1e-6)


def test_camera_directions_match_carla_forward_right_up_vectors():
    """The real proof for fact 5 and the yaw sign: build a CARLA transform
    from each camera's real extrinsics, ask CARLA for its forward/right/up
    vectors, convert back to rig frame, and compare against the optical
    axes straight from the golden extrinsics CSV.
    """
    for _cam_key, sensor_name in ALPAMAYO_CAMERAS.items():
        row = _load_extrinsics_row(sensor_name)
        r_opt2rig = _quat_xyzw_to_matrix(row["qx"], row["qy"], row["qz"], row["qw"])
        t_rig = np.array([row["x"], row["y"], row["z"]])

        loc, (pitch, yaw, roll) = coords.camera_extrinsics_to_carla_transform(r_opt2rig, t_rig)
        transform = carla.Transform(
            carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
        )

        fwd = transform.rotation.get_forward_vector()
        right = transform.rotation.get_right_vector()
        up = transform.rotation.get_up_vector()

        fwd_rig = coords.carla_to_rig_xyz(np.array([fwd.x, fwd.y, fwd.z]))
        right_rig = coords.carla_to_rig_xyz(np.array([right.x, right.y, right.z]))
        up_rig = coords.carla_to_rig_xyz(np.array([up.x, up.y, up.z]))

        optical_z_rig = r_opt2rig[:, 2]  # forward
        optical_x_rig = r_opt2rig[:, 0]  # right (image right)
        optical_y_rig = r_opt2rig[:, 1]  # down (image down)

        assert np.allclose(fwd_rig, optical_z_rig, atol=1e-4), sensor_name
        assert np.allclose(right_rig, optical_x_rig, atol=1e-4), sensor_name
        assert np.allclose(up_rig, -optical_y_rig, atol=1e-4), sensor_name
