"""World-frame <-> ego-frame(t0) pose conversions.

The formula and operation order here are copied exactly from the upstream
loader (`third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py`,
lines 130-150) so that our output is bit-identical to what the model was
built against:

    t0_rot = Rotation.from_quat(quat[-1])
    t0_rot_inv = t0_rot.inv()
    xyz_local = t0_rot_inv.apply(world_xyz - world_xyz[-1])
    rot_local = (t0_rot_inv * Rotation.from_quat(quat)).as_matrix()

All arithmetic is kept in float64 until the final cast to float32, matching
the upstream code path (which only casts to float32 when moving into a
torch tensor via `.float()`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def world_to_ego_history(
    world_xyz: np.ndarray, world_quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world-frame pose history into the ego frame of its last entry.

    Args:
        world_xyz: `(N, 3)` float64 world positions, oldest first.
        world_quat_xyzw: `(N, 4)` float64 quaternions in scipy order
            (x, y, z, w), oldest first. The last entry is t0.

    Returns:
        `(xyz_local, rot_local)`: `xyz_local` is `(N, 3)` float32, `rot_local`
        is `(N, 3, 3)` float32, both expressed in the ego frame of the last
        (t0) pose.
    """
    t0_rot = Rotation.from_quat(world_quat_xyzw[-1])
    t0_rot_inv = t0_rot.inv()

    xyz_local = t0_rot_inv.apply(world_xyz - world_xyz[-1])
    rot_local = (t0_rot_inv * Rotation.from_quat(world_quat_xyzw)).as_matrix()

    return xyz_local.astype(np.float32), rot_local.astype(np.float32)


def ego_to_world(
    xyz_local: np.ndarray,
    rot_local: np.ndarray,
    t0_xyz: np.ndarray,
    t0_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert `world_to_ego_history`: ego-frame(t0) poses back to world frame.

    Args:
        xyz_local: `(N, 3)` positions in the ego frame of t0.
        rot_local: `(N, 3, 3)` rotation matrices in the ego frame of t0.
        t0_xyz: `(3,)` world position of t0.
        t0_quat_xyzw: `(4,)` world orientation of t0, scipy (x, y, z, w) order.

    Returns:
        `(world_xyz, world_quat_xyzw)`: `(N, 3)` float64 positions and
        `(N, 4)` float64 quaternions (scipy order) in world frame.
    """
    t0_rot = Rotation.from_quat(t0_quat_xyzw)

    world_xyz = t0_rot.apply(xyz_local) + t0_xyz
    world_quat_xyzw = (t0_rot * Rotation.from_matrix(rot_local)).as_quat()

    return world_xyz, world_quat_xyzw


def carla_pose_to_world_quat(*args: object, **kwargs: object) -> None:
    """Convert a CARLA actor transform into world xyz + scipy-order quaternion.

    NOT IMPLEMENTED HERE, ON PURPOSE. CARLA uses a left-handed coordinate
    system with degrees and a pitch/yaw/roll (UE4-style) rotation convention;
    converting that into the right-handed, quaternion-based world frame this
    module expects is the job of M1 (the CARLA sensor rig / world-frame
    bridge), not of this M2 adapter. Guessing at the handedness/axis-order
    conversion here would silently bake in an unverified assumption at the
    exact spot where getting it wrong is most dangerous. Raise instead.
    """
    raise NotImplementedError(
        "carla_pose_to_world_quat is intentionally unimplemented in M2. "
        "The CARLA -> world-frame conversion (left-handed, degrees, "
        "pitch/yaw/roll) belongs to M1 and must not be guessed at here."
    )
