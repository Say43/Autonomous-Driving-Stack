# M4 CARLA open-loop evidence

Status: **passed for the recorder milestone** at implementation commit
`935011692986da3d853739447cb4f97cb9a02bda`.

This milestone does not run Alpamayo and does not relax the separate M0 gate.
It proves that the local CARLA side can create schema-valid, frame-correlated,
reproducible traces and clean up its actors afterward.

The machine-readable record is `results/m4_evidence.json`. Large JSONL and PNG
artifacts remain local and gitignored; the evidence file records their exact
paths, hashes, and counts.

## Verified results

- `pytest`: 73 passed; `ruff check .`: all checks passed.
- Four-camera smoke run: 100 world ticks, 50 complete camera sets, four images
  per set, 200 PNG files total (512,824,691 bytes). All 200 files have the PNG
  magic signature; no JPEG data is mislabeled.
- Frame correlation uses CARLA's `image.frame`, not callback order. The initial
  capture occurred on frame 0 and the stable cadence was every second world
  tick, matching 10 Hz cameras against a 20 Hz world.
- Ground-truth run: 600 ticks; nearby actors were present in 313 frames (maximum
  four) and traffic lights in 352 frames (maximum three).
- Determinism run: two independent 100-tick runs with seed 1 produced identical
  serialized `TraceFrame` lines. The full-file hashes differ only because the
  `RunHeader.run_timestamp` is intentionally different; both frame payloads
  hash to `76c5717f8d42d1f9a71e2f47b15b6ade3e10cded689ba2cd35bc299e67339d0e`.
- After the camera run, CARLA reported zero vehicles, walkers, and sensors, and
  synchronous mode had been restored to `false`.

## Reproduction

Start CARLA directly on the target map. This avoids a transient two-map VRAM
peak from `client.load_world()` on the 6 GiB GPU:

```powershell
..\carla-0.9.16\pkg\CarlaUE4.exe /Game/Carla/Maps/Town01 `
  -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

Then run the recorder from the Python 3.12 simulation environment:

```powershell
.\.venv-sim\Scripts\python.exe scripts\run_open_loop.py `
  --out results\m4_smoke_png --ticks 100 --seed 1 --traffic 10 --purge-actors
```

The default four-camera rig is `configs/rig_provisional.yaml`. It is deliberately
named provisional: its geometry is sufficient for simulator and viewer work,
but it is not the PhysicalAI AV calibration. Replacing it with measured M1
intrinsics/extrinsics remains blocked on authenticated access to the gated
dataset.
