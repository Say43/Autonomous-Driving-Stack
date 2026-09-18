# Alpamayo Drive — 3D trace viewer

White, Tesla-inspired driving visualization with an elevated camera behind the ego.
Runs offline with no build step, npm packages, CDN, or simulator connection.
The 3D meshes use a perspective camera, near-plane clipping, depth-sorted faces,
shading and distance fog. They are scaled visual proxies for recorded bounding boxes,
not CARLA mesh assets or model-generated detections.

## Open

Double-click `viewer/index.html`. It starts with a clearly labelled synthetic demo.
Choose **Trace öffnen**, or drop `trace.jsonl` anywhere on the page.
`file://` uses the File API only; it never tries to fetch other local files.

An optional localhost-only server can open an existing run together with its summary:

```powershell
node scripts/serve_viewer.cjs --trace results/loop_town10_seed21_run6_speedbound/trace.jsonl
```

Open the printed URL. Add `&t=45.05` to inspect a specific simulation time.
The server exposes the viewer, the one selected trace, and its sibling summary only.
It binds to `127.0.0.1`, not the network. Stop it with Ctrl+C.

## Controls

- Drag the scene to orbit; scroll to zoom; ↺ resets the camera.
- **3D Heck / Von oben** changes the camera projection.
- Play, single-frame steps, scrubber, and ¼×–4× playback use recorded simulation time.
- Space plays/pauses, left/right arrow keys step. Focused form controls retain their own keys.
- The upper-right menu toggles details; small screens start with them collapsed.
- Playback pauses when the page becomes hidden and stops at the end.

## Data provenance and old logs

Vehicles, pedestrians, static objects, lanes and traffic lights are simulator ground truth.
Only the blue plan and reasoning originate from Alpamayo in a real model run. Demo and
transport-test workers are labelled separately. The model receives camera images; it
does not provide the object detections shown in this viewer.

New traces store static objects once in `RunHeader.environment_objects`, lane widths
on lane segments, and intervention/collision/offroad diagnostics on each frame.
All recorded objects within the display radius are considered; normal perspective
occlusion and distance fog still apply. Object counts include environment mesh components.
An old trace cannot supply missing parked vehicles, vegetation or other static meshes:
the viewer explicitly warns about incomplete coverage and never fabricates them.
Older lane widths default to 3.5 m for display only; lines approximate lane geometry,
not individually recorded road-marking paint.

Repeated plans stay anchored to `plan.frame_id` and that frame's original ego pose.
For new reference-corrected traces the anchor is its explicit `model_pose_world`,
while the vehicle mesh remains at `ego_pose_world`. Legacy recordings without the
new field retain their original actor-origin interpretation.
FLU-left is reflected into CARLA-right exactly once. Missing plan origins cause a visible
warning; those paths are not reanchored to an unrelated pose. Reasoning shows its actual
source frame and age; a path is hidden after its 6.4 s horizon expires.

Malformed/out-of-order frames are skipped and counted. Invalid files report an error
without destroying the current replay. Loading a new file stops playback, and a newer
file selection cannot be overwritten by an older asynchronous load.

## Regression checks

```powershell
node --test tests/viewer.test.cjs
```

Checks cover plan anchoring, FLU reflection, sparse/repeated plans, missing origins,
malformed frames, and finite 3D rendering in both camera projections. The release was
also rendered in a headless Edge instance using a real 1,200-frame run and a new
trace with static environment objects. No user's browser profile is required.
