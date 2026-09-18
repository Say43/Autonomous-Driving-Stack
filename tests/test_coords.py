"""Non-oracle tests for `src/acarla/sim/coords.py`.

Anchors the fixed conventions against the golden calibration CSV directly
(no carla/physical_ai_av import needed). The CARLA-Euler convention itself
and the camera-direction proof are validated against the real CARLA client
in `tests/oracle/test_coords_carla_oracle.py` -- this file only checks
internal consistency and the parts derivable without CARLA.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from acarla.sim import coords

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden" / "m2_reference"

# Calibration data extracted from the PhysicalAI-AV dataset is licensed for internal
# use only (NVIDIA AV Dataset License, section 4.6) and is therefore not part of the
# public repository; regenerate it with the notebooks/scripts if you hold a licence.
pytestmark = pytest.mark.skipif(
    not (GOLDEN_DIR / "calibration_sensor_extrinsics.csv").exists(),
    reason="private PhysicalAI-AV calibration CSVs not present",
)


def _load_extrinsics_row(sensor_name: str) -> dict:
    with open(GOLDEN_DIR / "calibration_sensor_extrinsics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["sensor_name"] == sensor_name:
                return {k: (float(v) if k != "sensor_name" else v) for k, v in row.items()}
    raise KeyError(sensor_name)


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Local, dependency-free quaternion -> rotation matrix (scipy xyzw order),
    # used only to anchor R_OPTICAL_TO_FLU against the golden CSV without
    # requiring scipy in this environment's import graph for coords.py itself.
    x, y, z, w = qx, qy, qz, qw
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def test_rig_to_carla_xyz_mirrors_y():
    xyz = np.array([1.0, 2.0, 3.0])
    out = coords.rig_to_carla_xyz(xyz)
    assert np.allclose(out, [1.0, -2.0, 3.0])
    assert np.allclose(coords.carla_to_rig_xyz(out), xyz)


def test_rig_to_carla_xyz_batched():
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    out = coords.rig_to_carla_xyz(xyz)
    assert np.allclose(out, [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])


def test_rig_rotation_to_carla_rotation_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        # Random orthonormal matrix via QR.
        a = rng.standard_normal((3, 3))
        q, _ = np.linalg.qr(a)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        carla_r = coords.rig_rotation_to_carla_rotation(q)
        back = coords.carla_rotation_to_rig_rotation(carla_r)
        assert np.allclose(back, q, atol=1e-10)
        # Orthonormality and determinant are preserved by conjugation with a
        # reflection: det(M R M) = det(M)^2 det(R) = det(R).
        assert np.isclose(np.linalg.det(carla_r), np.linalg.det(q))
        assert np.allclose(carla_r @ carla_r.T, np.eye(3), atol=1e-10)


def test_r_optical_to_flu_is_orthonormal_rotation():
    r = coords.R_OPTICAL_TO_FLU
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1.0)


def test_r_optical_to_flu_matches_front_wide_extrinsics():
    """Fact 3 from the M1 brief, anchored to the actual golden numbers:
    for camera_front_wide_120fov the extrinsic rotation matrix's z column
    (optical forward) is close to rig +x, its x column (optical right) is
    close to rig -y, and its y column (optical down) is close to rig -z.
    R_OPTICAL_TO_FLU is exactly that nominal mapping; front_wide's actual
    extrinsics should be close to it (small residual rotation only).
    """
    row = _load_extrinsics_row("camera_front_wide_120fov")
    r_actual = _quat_xyzw_to_matrix(row["qx"], row["qy"], row["qz"], row["qw"])
    assert np.allclose(r_actual, coords.R_OPTICAL_TO_FLU, atol=0.02)


@pytest.mark.parametrize(
    "pitch,yaw,roll",
    [(0.0, 0.0, 0.0), (0.0, 90.0, 0.0), (90.0, 0.0, 0.0), (0.0, 0.0, 90.0), (10.0, 20.0, 30.0)],
)
def test_carla_euler_to_matrix_orthonormal_det_plus_one(pitch, yaw, roll):
    r = coords.carla_euler_to_matrix(pitch, yaw, roll)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-10)


def test_carla_euler_matrix_roundtrip_via_matrix_compare():
    rng = np.random.default_rng(1)
    for _ in range(50):
        pitch = rng.uniform(-89, 89)
        yaw = rng.uniform(-180, 180)
        roll = rng.uniform(-89, 89)
        r = coords.carla_euler_to_matrix(pitch, yaw, roll)
        p2, y2, r2 = coords.matrix_to_carla_euler(r)
        r_back = coords.carla_euler_to_matrix(p2, y2, r2)
        assert np.allclose(r, r_back, atol=1e-8)


def test_carla_euler_identity():
    r = coords.carla_euler_to_matrix(0.0, 0.0, 0.0)
    assert np.allclose(r, np.eye(3))


def test_camera_extrinsics_to_carla_transform_front_wide_yaw_near_zero():
    row = _load_extrinsics_row("camera_front_wide_120fov")
    r_opt2rig = _quat_xyzw_to_matrix(row["qx"], row["qy"], row["qz"], row["qw"])
    t_rig = np.array([row["x"], row["y"], row["z"]])
    loc, (pitch, yaw, roll) = coords.camera_extrinsics_to_carla_transform(r_opt2rig, t_rig)
    assert abs(yaw) < 2.0
    assert np.allclose(loc, [row["x"], -row["y"], row["z"]])


def test_camera_extrinsics_to_carla_transform_cross_left_right_opposite_sign():
    row_l = _load_extrinsics_row("camera_cross_left_120fov")
    row_r = _load_extrinsics_row("camera_cross_right_120fov")
    r_l = _quat_xyzw_to_matrix(row_l["qx"], row_l["qy"], row_l["qz"], row_l["qw"])
    r_r = _quat_xyzw_to_matrix(row_r["qx"], row_r["qy"], row_r["qz"], row_r["qw"])
    _, (_, yaw_l, _) = coords.camera_extrinsics_to_carla_transform(
        r_l, np.array([row_l["x"], row_l["y"], row_l["z"]])
    )
    _, (_, yaw_r, _) = coords.camera_extrinsics_to_carla_transform(
        r_r, np.array([row_r["x"], row_r["y"], row_r["z"]])
    )
    # cross_left looks left (rig +y), CARLA yaw should be negative;
    # cross_right looks right (rig -y), CARLA yaw should be positive.
    assert yaw_l < -50
    assert yaw_r > 50
    assert np.sign(yaw_l) != np.sign(yaw_r)
    assert 50 < abs(yaw_l) < 80
    assert 50 < abs(yaw_r) < 80


def test_camera_extrinsics_result_is_orthonormal_rotation():
    row = _load_extrinsics_row("camera_front_tele_30fov")
    r_opt2rig = _quat_xyzw_to_matrix(row["qx"], row["qy"], row["qz"], row["qw"])
    t_rig = np.array([row["x"], row["y"], row["z"]])
    _, (pitch, yaw, roll) = coords.camera_extrinsics_to_carla_transform(r_opt2rig, t_rig)
    r = coords.carla_euler_to_matrix(pitch, yaw, roll)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-8)
