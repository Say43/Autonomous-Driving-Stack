# Closed-loop hardening and 3D viewer — September 2026

This change addresses observed integration/controller failures from run 6. It does
not retrain Alpamayo, establish real-time operation, or establish collision-free
model driving across maps and seeds. The last completed real-model evaluation
is still run 6 until a new Kaggle-backed run is recorded.

## Changes

- A simulator-ground-truth supervisor checks the actual vehicle footprint along
  both the model path and the path implied by the applied steering command.
  Lookahead covers reaction time and braking distance. It checks driving-lane
  containment and oriented obstacle boxes, including parked/static meshes.
- Moving obstacles are checked under constant-velocity and hard-braking
  predictions. Bounding-box centers include the actor-local offsets. Overhead
  geometry and flat road details below the configured clearance are excluded.
- The controller caps target speed by lateral acceleration in curves, resets
  the PID after overrides, and immediately invalidates a plan after worker errors.
  AEB uses actual ego dimensions and accounts for oncoming closing velocity.
- The recorder caches lane geometry and static objects once. New headers contain
  `environment_objects`; lane segments include their widths. Static environment
  boxes are already in world coordinates and are not transformed twice. See the
  [CARLA EnvironmentObject API](https://carla.readthedocs.io/en/latest/python_api/#carla.EnvironmentObject).
- Per-frame diagnostics expose AEB, supervisor reason, speed, warmup, collision,
  and offroad state. `footprint_offroad_ticks` measures footprint departures
  independently of whether the supervisor is enabled.
- The default run stops on collision. Diagnostic continuation requires
  `--continue-on-collision`. Nonempty output directories are rejected; partial
  counters and errors survive an aborted run in `summary.json`.
- `evaluation.safe_completion` requires completion, progress, plans, no collision,
  no center/footprint departure, and no controller/execution fault.
  `unassisted_model_pass` additionally requires actual Alpamayo and zero rescue
  interventions. A fake worker requires `--allow-fake-worker` and always records
  `alpamayo_used: false`. Sensor warmup is separate from controller faults.
- The white 3D viewer has a rear camera, shaded vehicle/pedestrian/scene meshes,
  orbit/zoom/top-down controls, simulation-time playback, and visible provenance.
  It fixes a real replay bug: repeated plans now use the original inference pose
  from `plan.frame_id`, rather than moving with each subsequent ego pose.
  Old traces explicitly show missing static-object coverage.

## Evidence and reproduction

The changes were developed in an isolated clone. Existing run-6 data and the original
worktree were preserved during development. Validation uses a separate CARLA instance
on port 2100 / Traffic Manager 8100, with no Hugging Face writes or Kaggle job submission.

Final local end-to-end test (`results/quality_smoke_final/summary.json`, 18 September;
tracked copy: `results/quality_smoke_final_summary.json`):

| Measurement | Result |
|---|---:|
| Simulation | 200/200 ticks, 10 s |
| Worker | Local straight-plan test worker, **not Alpamayo** |
| Plans / distance | 8 / 26.27 m |
| Collisions / offroad center / offroad footprint / controller faults | 0 / 0 / 0 / 0 |
| AEB / supervisor ticks | 0 / 0 |
| Static environment components recorded | 66,120 |

The executable source for this run is commit `2cdb0f58ed7e8f87acf40b0a66bc06403c092733`.
Only documentation differed in the checkout. The raw recorder reported
`git_sha: unknown` because Git rejected the isolated checkout's different Windows
owner when invoked by the simulator environment. Raw artifacts were not rewritten:
`results/quality_validation_manifest.json` records the externally verified source
and SHA-256 hashes. This ownership issue does not occur in the original user-owned
checkout; no global Git trust exception was added.

The final controlled safety tests (`results/quality_safety_scenarios_final.json`,
source `2cdb0f58ed7e8f87acf40b0a66bc06403c092733`, 18 September) each completed
300 ticks: a parked-car test stopped after 11.32 m; the deliberately sidewalk-bound
plan stopped after 5.95 m. Both had zero collisions, center-offroad ticks **and
footprint-offroad ticks**, with explicit interventions. The first parked-car test fixture was invalid
because it read the actor snapshot before the first tick; it is preserved in
`quality_safety_scenarios.json` and is not counted as passing evidence. The fixture
now verifies that its obstacle actually lies 19 m ahead.

Final automated checks: **178 Python tests passed, 2 skipped; 7 viewer tests passed;
4 additional coordinate-oracle tests passed against the installed CARLA API**.
The two default-environment skips are the optional CARLA and PhysicalAI oracle
dependencies; the CARLA oracle was separately run in `.venv-sim`. Changed Python
files pass Ruff. The new tests cover bbox offsets, oncoming/crossing traffic,
static obstacles, overhead objects, manhole covers, footprint containment, curved
speed targets, worker-plan invalidation and honest pass/fail classification.

The existing remote queue heartbeat was checked read-only on 18 September:
the worker reported `state: exited`, with a 38.5-hour-old heartbeat. No new
real-model run was queued against that inactive worker. Restarting the existing
`says43/alpamayo-loop-worker` in Kaggle with its attached Hugging Face secret is
required before the next real Alpamayo test. The notebook and secret bindings were
not modified by this work.

The off-policy replay audit (`results/quality_audit_run6.json`) queries the original
recorded poses against the real CARLA map. Its first pass exposed a false positive
on a flat manhole cover, which was fixed and regression-tested. An audit of recorded
states is **not** a new closed-loop result: subsequent vehicle/model states would
change after the first intervention.

Reproduce unit and viewer checks:

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
node --test tests/viewer.test.cjs
```

Reproduce controlled scenarios on a **dedicated** CARLA server:

```powershell
.venv-sim\Scripts\python.exe scripts/run_safety_scenarios.py --port 2100 --out results/new_safety_scenarios.json
```

Reproduce the full transport/recorder smoke in two terminals, with new output paths:

```powershell
.venv\Scripts\python.exe scripts/fake_loop_worker.py --repo dir:results/new_quality_queue --speed 4 --max-jobs 8
.venv-sim\Scripts\python.exe scripts/run_closed_loop_alpamayo.py --port 2100 --tm-port 8100 --repo dir:results/new_quality_queue --out results/new_quality_smoke --seed 21 --ticks 200 --traffic 0 --allow-fake-worker --stop-worker-at-end
```

The viewer was rendered in a separate headless Edge profile with the original
1,200-frame run at t=45.05 s, the new trace including static objects, and the offline
demo. No user's browser profile was modified. See [viewer usage](../viewer/README.md).

## Remaining limits

The supervisor is a conservative simulation aid. Bounding boxes approximate meshes,
vehicle dynamics and obstacle motion; difficult cases can still cause false stops or
unmodelled collisions. Stopping indefinitely is not a successful drive: the evaluator
also requires progress. It does not plan an overtaking route, guarantee traffic-rule
compliance, or replace a complete safety case.

Alpamayo's text/path contradictions, CARLA image-domain gap, and camera-rig offset
remain model/calibration questions. The code does not silently mirror a trajectory
based on English reasoning. The new protection must still be tested in fresh
real-model runs across multiple seeds/maps. Existing results cannot prove those runs.

The viewer's Tesla-inspired styling is an independent implementation with simplified
meshes. Objects that were never recorded cannot be reconstructed from old traces.

## First real-model run on the hardened stack (18 September, run 7)

The Kaggle-backed run exposed four defects that the fake-worker smoke and the two
controlled scenarios could not reach, all fixed the same day with regression tests
(`f4a33d9`, `167260d`, `952a730`, `208267d`), scenarios still 2/2:

| Attempt | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | crash after plan 0 | `print` of the reasoning text (U+2011) on a cp1252 console | stdout/stderr reconfigured to UTF-8 with replacement |
| 2 | deadlock from t=19 s, `static.poles` 2.4 m ahead | street light: mast at the kerb, 3.8 m arm over the lane, one 7.8 m tall box treated as a wall (199 of 880 poles, plus tree canopies) | boxes of poles/lights/signs/vegetation that are both tall (>= 2.5 m) and wide (>= 1 m) are skipped; kerb masts remain covered by lane containment |
| 3 | deadlock from t=32 s at a green light | strict point-in-lane-polygon test: ego 0.6 m right of centre, one footprint edge point 1 cm outside the lane line, no shoulder polygon there; later the same inside the junction (polygon gaps between turning lanes) | `groundtruth.make_drivable_check`: inside a driving-lane polygon OR within half a lane width + 0.5 m of the nearest driving lane, + 1.5 m inside junctions; shared by the loop, the scenarios and the replay audit |
| 4 | stop from t=27 s | **genuine**: after "Change lane to the left" the ego centre sits 0.5 m beyond the left lane edge, the left-front corner heads onto the median shoulder while the model keeps planning straight ahead | none -- this is the intervention the supervisor exists for |

Replay audit of run 6 after the fixes (`results/quality_audit_run6_v2.json`): 0 static
false positives (8 before), remaining brake frames are the stopped Lincoln (19) and lane
departure (54).

Result of run 7 (`results/loop_town10_seed21_run7_hardened`, source `952a730`, Alpamayo on
Kaggle, Town10HD_Opt seed 21, 1 s replans): 60 s complete, 58 plans, **0 collisions**,
92.9 m driven, max 9.4 m/s, 13 AEB ticks, **856 supervisor ticks**, 739 centre-offroad ticks,
0 footprint-offroad ticks. `evaluation`: `safe_completion: false` (offroad),
`unassisted_model_pass: false`, `assisted: true`. The ego drove 27 s, then stood 33 s at the
median with the supervisor holding the brake while the model kept planning straight ahead
("Keep lane since the lane is clear"). Compared with run 6 (same seed, no supervisor):
3 collision ticks became 0, 316 m became 93 m. That is the honest trade: the supervisor
converts the model's road departures from crashes into standstills, it does not make the
model drive better.

Lesson for the next iteration: every supervisor rule must be checked for the standstill
case. A false "leaves driving lanes" at 0 m/s is a permanent deadlock, because the model
replans from the same pose and never gets a chance to correct.
