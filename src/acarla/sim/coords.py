"""Coordinate-frame conversions between the Alpamayo/PhysicalAI-AV rig frame
and CARLA's world/actor frame.

This module is pure numpy: it never imports `carla` or `physical_ai_av`, so
it can be exercised in the 3.13 test environment. Every convention encoded
here is checked against a real oracle in ``tests/oracle`` (CARLA's own
``carla.Transform.get_matrix()``/``get_forward_vector()`` etc.), which only
runs in ``.venv-sim``.

Three frames are in play:

- **Rig frame** (Alpamayo/PhysicalAI-AV): x forward, y left, z up,
  right-handed ("FLU"). This is the frame the golden calibration CSVs are
  expressed in.
- **CARLA world/actor frame**: x forward, y right, z up, left-handed.
  Rotations are given as (pitch, yaw, roll) in degrees.
- **Camera optical frame** (OpenCV convention, used by the extrinsics CSV
  and by `physical_ai_av`'s camera models): x right (in image), y down (in
  image), z forward (viewing direction), right-handed.

Rig <-> CARLA world only differs by mirroring the y axis (``y -> -y``);
x (forward) and z (up) are shared. That single mirror, applied consistently,
is `rig_to_carla_xyz` / `carla_to_rig_xyz` for points and
`rig_rotation_to_carla_rotation` / `carla_rotation_to_rig_rotation` for
rotation matrices that are already expressed *directly* in one of these two
axis systems (e.g. an ego rotation matrix). It must **not** be used to
convert a rotation matrix whose domain basis is something else (such as a
camera's own local axes) -- see `camera_extrinsics_to_carla_transform` for
that case, which builds the result axis-by-axis instead.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Rig <-> CARLA axis mirror
# ---------------------------------------------------------------------------

_MIRROR_Y = np.diag(np.array([1.0, -1.0, 1.0]))
"""The single fixed operation relating rig (FLU, right-handed) and CARLA
world (FRU, left-handed) axes: negate y, keep x and z."""


def rig_to_carla_xyz(xyz: np.ndarray) -> np.ndarray:
    """Convert a point/vector from rig frame (x fwd, y left, z up) to CARLA
    world frame (x fwd, y right, z up). Just negates y."""
    xyz = np.asarray(xyz, dtype=float)
    out = xyz.copy()
    out[..., 1] = -out[..., 1]
    return out


def carla_to_rig_xyz(xyz: np.ndarray) -> np.ndarray:
    """Inverse of `rig_to_carla_xyz`. The mirror is its own inverse."""
    return rig_to_carla_xyz(xyz)


def rig_rotation_to_carla_rotation(r_rig: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix expressed directly in rig-frame axes
    (domain and codomain both FLU) to the equivalent matrix expressed
    directly in CARLA world-frame axes (domain and codomain both FRU).

    Mathematically ``M @ R @ M`` with ``M = diag(1, -1, 1)``, i.e. mirror
    the input basis, apply the rotation, mirror back.
    """
    r_rig = np.asarray(r_rig, dtype=float)
    return _MIRROR_Y @ r_rig @ _MIRROR_Y


def carla_rotation_to_rig_rotation(r_carla: np.ndarray) -> np.ndarray:
    """Inverse of `rig_rotation_to_carla_rotation`. The conjugation by the
    involutive mirror `_MIRROR_Y` is its own inverse."""
    return rig_rotation_to_carla_rotation(r_carla)


# ---------------------------------------------------------------------------
# Optical (OpenCV) <-> rig FLU
# ---------------------------------------------------------------------------

R_OPTICAL_TO_FLU = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
"""Fixed 3x3 matrix mapping the *nominal* optical axes (x right in image,
y down in image, z forward / viewing direction) onto rig FLU axes (x
forward, y left, z up), i.e. the matrix such that
``rig_vector = R_OPTICAL_TO_FLU @ optical_vector`` for a camera mounted
looking straight forward with no roll.

Derived from fact 3 in the M1 brief: for `camera_front_wide_120fov` the
extrinsic rotation matrix's z column (optical forward) is approximately
rig +x, its x column (optical right) is approximately rig -y, and its y
column (optical down) is approximately rig -z. This matrix is exactly that
nominal mapping; the real extrinsics for each camera are a small rotation
away from it (verified in tests/test_coords.py against the golden CSV).
"""


# ---------------------------------------------------------------------------
# CARLA Euler <-> rotation matrix
#
# Convention verified against carla.Transform(...).get_matrix() in
# tests/oracle/test_coords_carla_oracle.py (200 random (pitch, yaw, roll)
# triples in [-180, 180], atol=1e-6) and against get_forward_vector() /
# get_right_vector() / get_up_vector() for the four Alpamayo cameras.
#
# Finding: CARLA's get_matrix() returns an orthonormal matrix with
# det == +1. This is *not* a general right-handed rotation matrix acting on
# a right-handed frame -- CARLA's world axes (x fwd, y right, z up) are
# themselves left-handed, so a "det +1, orthonormal" matrix expressed in
# that left-handed basis is CARLA's own self-consistent representation of a
# physical rotation; there is no separate sign flip hidden in the matrix
# itself. The matrix's columns are, in order, the world-frame forward,
# right, and up unit vectors of the rotated frame -- confirmed by comparing
# against get_forward_vector()/get_right_vector()/get_up_vector().
#
# The element formula (matches get_matrix() to 1e-6 for 200 random angles):
#
#   cy, sy = cos(yaw), sin(yaw)
#   cp, sp = cos(pitch), sin(pitch)
#   cr, sr = cos(roll), sin(roll)
#
#   R = [[ cp*cy,  cy*sp*sr - sy*cr,  -cy*sp*cr - sy*sr ],
#        [ sy*cp,  sy*sp*sr + cy*cr,  -sy*sp*cr + cy*sr ],
#        [ sp,     -cp*sr,             cp*cr             ]]
#
# i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll) applied in CARLA's left-handed
# axis labelling (this is the standard aerospace yaw-pitch-roll product,
# carried over unchanged into the left-handed frame).
# ---------------------------------------------------------------------------


def carla_euler_to_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Build the 3x3 rotation matrix CARLA's ``carla.Transform.get_matrix()``
    (upper-left 3x3 block) returns for the given (pitch, yaw, roll) in
    degrees. See module-level comment above for the verified convention.
    """
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    roll = np.radians(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )


def matrix_to_carla_euler(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `carla_euler_to_matrix`: recover (pitch, yaw, roll) in
    degrees from a rotation matrix in CARLA's convention.

    Euler angles are not unique (gimbal lock at pitch = +-90 deg); this
    picks the branch with pitch in [-90, 90]. Callers that need a
    roundtrip-stable comparison should compare the resulting matrices, not
    the angles themselves (see the oracle roundtrip test).
    """
    r = np.asarray(r, dtype=float)
    pitch = np.arcsin(np.clip(r[2, 0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        yaw = np.arctan2(r[1, 0], r[0, 0])
        roll = np.arctan2(-r[2, 1], r[2, 2])
    else:
        # Gimbal lock: pitch = +-90 deg, yaw and roll become degenerate.
        # Pick roll = 0 and solve for yaw from the remaining block.
        yaw = np.arctan2(-r[0, 1], r[1, 1])
        roll = 0.0
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


# ---------------------------------------------------------------------------
# Full sensor extrinsics -> CARLA camera transform
# ---------------------------------------------------------------------------


def camera_extrinsics_to_carla_transform(
    r_optical_to_rig: np.ndarray, t_rig: np.ndarray
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Convert a camera's rig-frame extrinsics (rotation optical->rig, and
    translation in rig frame) into a CARLA camera transform: a location in
    CARLA world/actor coordinates and a (pitch, yaw, roll) in degrees for a
    ``sensor.camera.rgb`` that looks along its own +x, whose image-x maps to
    CARLA +y (right) and image-y maps to CARLA -z (down) -- fact 5 in the M1
    brief, proven against the oracle in
    tests/oracle/test_coords_carla_oracle.py.

    Built axis-by-axis rather than via `rig_rotation_to_carla_rotation`:
    that helper assumes its input matrix's domain basis is already rig FLU,
    but here the domain is the camera's own local axes (optical / CARLA
    sensor-local), a different basis whose y axis does not correspond to
    rig y. Instead: take the camera's physical forward/right/up direction
    vectors (from the optical rotation matrix, in rig coordinates), convert
    each to CARLA world coordinates with the same mirror used for points,
    and stack them as the columns of the CARLA rotation matrix -- which is
    exactly what `carla.Transform.get_matrix()`'s columns mean (see the
    convention note above `carla_euler_to_matrix`).
    """
    r_optical_to_rig = np.asarray(r_optical_to_rig, dtype=float)
    t_rig = np.asarray(t_rig, dtype=float)

    forward_rig = r_optical_to_rig[:, 2]  # optical z = viewing direction
    right_rig = r_optical_to_rig[:, 0]  # optical x = image right
    up_rig = -r_optical_to_rig[:, 1]  # optical y = image down, so -y = up

    forward_carla = rig_to_carla_xyz(forward_rig)
    right_carla = rig_to_carla_xyz(right_rig)
    up_carla = rig_to_carla_xyz(up_rig)

    r_world = np.column_stack([forward_carla, right_carla, up_carla])
    pitch, yaw, roll = matrix_to_carla_euler(r_world)

    location_carla = rig_to_carla_xyz(t_rig)
    return location_carla, (pitch, yaw, roll)
