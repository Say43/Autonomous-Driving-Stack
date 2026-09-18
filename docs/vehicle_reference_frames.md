# Vehicle reference-point correction

## What was inconsistent

The camera attachment applied a configured rig-to-actor offset plus an optional
mesh-clearance shift. However, ego history and future trajectory anchoring used
the unshifted CARLA actor pose. Pure Pursuit also treated that actor origin as its
bicycle rear axle. An axis reflection test does not detect these translation errors.

## Convention now used

All offsets below are in CARLA-local coordinates and metres. Let `A` be the actor
translation and `R` its rotation. Let `b` be the configured rear-axle/rig offset and
`s` the mesh-clearance shift. The model's virtual camera-rig origin is:

`M = A + R (b + s)`

A camera at calibrated rig position `c` is attached at `b + s + c`. Therefore its
world position is exactly `M + R c`. **Every** historical ego sample is transformed
this way, before conversion to ego-relative FLU coordinates. A single constant
offset added after history construction would miss the rotational lever arm.

Model predictions remain predictions of that virtual origin. For each predicted
position `P_i` and predicted body rotation `R_i`, the corresponding paths are:

- Actor: `P_i - R_i (b + s)` — footprint supervision.
- Rear axle: `P_i - R_i s` — Pure Pursuit and progress-based speed targeting.

The current rear-axle pose is `A + R b`. The supervisor's steering rollout pivots
at this point and transforms back to the actor before placing its footprint.
Predicted body rotations, not the tangent of the offset actor path, orient the
planned footprint. The steering and speed limits are otherwise unchanged.

## Recordings and compatibility

New traces preserve `ego_pose_world` as the actor pose and add `model_pose_world`.
Headers and `rig_used.json` record the offsets in `vehicle_geometry`. Offline
packet construction, viewer plan anchoring and replay audits consume this explicit
convention. Repeated plans retain the model pose at their original inference frame.
The displayed vehicle remains at the actor pose.

Legacy traces lacking the field retain their original actor-based interpretation.
Do not retroactively relabel their trajectories or count them as corrected runs.
The worker wire format is unchanged; it still receives ego-relative histories and
returns ego-relative predictions. The local configuration hash includes a new
reference-frame revision.

## Verification and limits

Public synthetic tests cover rotated camera attachments, turning ego histories,
per-waypoint rotation of offsets, analytic circular rear-axle tracking under
multiple world headings, the supervisor's physical pivot/body orientation, real
offline packet-builder execution, trace serialization and old/new viewer anchoring.
No private calibration values are used in these tests.

Local verification on 18 September 2026: **161 Python tests passed, 33 skipped**
(optional/private-input tests); **8 viewer tests passed**. Ruff passed on all
changed Python files. The CARLA runner also imports and parses `--help` in the
installed simulator environment. No new CARLA driving or remote-model run is
included in these counts.

This is a **coordinate-consistency fix, not proof of better Alpamayo lane keeping**.
The rear-axle offset still comes from the existing rig-origin hypothesis; it is not
a fresh measurement of CARLA wheel positions. Moving the cameras for mesh clearance
still changes their relation to the physical axle/body and is a domain-transfer
compromise. The speed feedback remains the actor's reported speed. Hardware-specific
steering response and model text/trajectory contradictions are not resolved here.

A new real-model closed-loop comparison is required. Keep map, seed, model sampling,
rig, controller limits and traffic setup fixed; report model-path lane error and
vehicle-to-path tracking error separately, plus collisions, strict and tolerant
offroad metrics, progress and intervention duration. Existing run 7 is a baseline,
not evidence that this correction passed.
