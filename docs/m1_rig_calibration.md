# M1 -- Rig calibration and coordinate transforms (method)

This note describes how the camera rig of the PhysicalAI-AV reference vehicle is
transferred into CARLA. The calibration values themselves (camera poses, F-Theta
polynomials, vehicle dimensions) are extracted from the NVIDIA PhysicalAI-AV dataset,
whose licence permits internal use only; they are therefore **not part of this
repository**. `scripts/calibrate_rig.py` regenerates `configs/rig_alpamayo.yaml` from the
dataset's calibration tables for anyone holding a dataset licence
(see `docs/data_and_licensing.md`). Every statement below is backed by an oracle test that
runs against the real `physical_ai_av` camera model and the real CARLA client when those
inputs are present (`tests/test_coords.py`, `tests/test_lens.py`,
`tests/oracle/test_coords_carla_oracle.py`, `tests/oracle/test_lens_physical_ai_oracle.py`).

## Frames

| Frame | Axes | Handedness | Evidence |
|---|---|---|---|
| Rig (Alpamayo / PhysicalAI-AV) | x forward, y left, z up ("FLU") | right-handed | the left cross camera has a positive rig-y coordinate, the right one a negative one; the upstream unicycle action space integrates positive curvature towards +y |
| CARLA world / actor | x forward, y right, z up | left-handed | `carla.Transform` |
| Camera optical (OpenCV) | x right, y down, z forward | right-handed | extrinsics quaternions; upstream `camera_models.py` docstrings |

Rig and CARLA world differ only by a reflection of the y axis
(`x, y, z -> x, -y, z`), applied to points (`coords.rig_to_carla_xyz`) and, for rotation
matrices expressed in one of the two bases, as `M @ R @ M` with `M = diag(1, -1, 1)`
(`coords.rig_rotation_to_carla_rotation`).

## CARLA Euler convention (proven against the oracle)

The upper-left 3x3 block of `carla.Transform.get_matrix()` is orthonormal with determinant
+1 for every (pitch, yaw, roll) tested (200 random triples, oracle test). CARLA's axis
labelling is itself left-handed, so a proper rotation matrix in that basis is CARLA's
self-consistent representation of a physical rotation; there is no hidden reflection
inside the matrix. The verified element formula is `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`,
whose columns are the world-frame forward/right/up vectors of the rotated frame
(confirmed against `carla.Rotation.get_forward_vector()/get_right_vector()/get_up_vector()`).
Implementation: `src/acarla/sim/coords.py::carla_euler_to_matrix` / `matrix_to_carla_euler`.

## Camera direction proof

For each camera, `camera_extrinsics_to_carla_transform` takes the physical
forward/right/up directions from the extrinsics (forward = optical z column, right =
optical x column, up = -optical y column), mirrors each into CARLA world coordinates and
stacks them as the columns of the CARLA rotation matrix. The oracle test round-trips this
through a real `carla.Transform`, asks CARLA for its direction vectors, converts back into
rig coordinates and compares with the optical axes from the calibration table
(`atol = 1e-4`, all four cameras). This is the concrete proof that CARLA image-x
corresponds to CARLA +y and image-y to CARLA -z, and of the yaw sign convention: the two
cross cameras receive yaws of opposite sign, the front cameras a yaw close to zero.

## F-Theta: reprojection instead of FOV matching

CARLA's `sensor.camera.rgb` is a pinhole camera; the dataset cameras are F-Theta fisheyes
(`r = fw_poly(theta)`, `theta = bw_poly(r)`). A pinhole camera cannot reproduce that radial
distortion, so instead of matching a single field-of-view number the pipeline renders a
*wider* pinhole image and reprojects it pixel by pixel into the F-Theta layout
(`src/acarla/adapter/lens.py::build_remap` / `apply_remap`): for every target pixel the
ray is obtained from `FThetaModel.pixel2ray`, the source pixel from
`PinholeModel.ray2pixel`, followed by bilinear resampling.

`required_pinhole_hfov_deg` computes the smallest pinhole horizontal FOV that still
contains every F-Theta pixel. It treats horizontal and vertical half-angles independently:
on a 16:9 sensor the corners of a 120-degree-class fisheye are reached mostly
horizontally, and an isotropic treatment would grossly overestimate the required FOV.
With the per-axis treatment all three wide cameras stay below the 150-degree cut-off
above which a single pinhole render is no longer useful; the tele camera needs well under
40 degrees.

## Open points carried into M4

- **Attachment origin.** CARLA attaches sensors relative to the vehicle actor's origin
  (ground level under the bounding-box centre); the dataset extrinsics are relative to a
  rig origin whose definition the dataset does not state explicitly. The working
  hypothesis -- rear axle at ground height -- is flagged as such in the generated config
  (`rig_origin_note`, `carla_attach_offset_open_issue: true`) and resolved empirically in
  M4 (`docs/m4_real_rig.md`).
- **Ego mask.** The offline intrinsics tables may carry an ego-vehicle mask; this has not
  been used.
