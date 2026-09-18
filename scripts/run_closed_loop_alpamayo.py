#!/usr/bin/env python3
"""M5: Alpamayo closes the CARLA loop -- in slow motion.

Every replan: CARLA (paused, synchronous) -> 4 cameras x 4 frames through the
calibrated F-Theta rig + 16 ego poses -> job on the Hugging Face queue ->
Kaggle worker runs Alpamayo-1.5 -> plan comes back -> the verified M5
controller (Pure Pursuit + PID + safety gate) drives it for `--replan-interval`
seconds of *simulated* time -> next image. The world does not tick while a
plan is in flight, so the ~30-60 s wall-clock round trip never becomes plan
age. Alpamayo therefore sees the consequences of its own previous plan: a real
closed loop, just not real time.

Prerequisites: CARLA server running on the target map; a Hugging Face login on
this machine (`hf auth login`, write access to the private queue repo); the
worker notebook `notebooks/kaggle_loop_worker.ipynb` started in the Kaggle
browser (Save & Run All with the `huggingface` secret attached). If the worker
is not up yet, this script waits and prints the worker heartbeat; the job stays
queued and is picked up as soon as the worker loads. Run from `.venv-sim`.

Usage:
    python scripts/run_closed_loop_alpamayo.py --out results/loop_town10_seed21 \\
        --seed 21 --map Town10HD_Opt --ticks 1200 --replan-interval 1.0 --traffic 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import carla
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# Model reasoning text is arbitrary Unicode; a redirected Windows console is cp1252
# (run 7 died printing U+2011 after the first plan). Never let logging kill a run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.adapter.lens import PinholeModel, apply_remap, build_remap  # noqa: E402
from acarla.adapter.packet import build_sensor_packet  # noqa: E402
from acarla.control import ControllerConfig, TrajectoryController  # noqa: E402
from acarla.control.aeb import AebConfig, apply_aeb, check_forward_corridor  # noqa: E402
from acarla.control.evaluation import evaluate_run  # noqa: E402
from acarla.control.supervisor import (  # noqa: E402
    SupervisorDecision,
    check_environment,
    footprint_is_drivable,
)
from acarla.loop.codec import decode_plan, encode_job  # noqa: E402
from acarla.loop.transport import DEFAULT_REPO, LoopQueue, open_queue  # noqa: E402
from acarla.model.inputs import sensor_packet_to_model_inputs  # noqa: E402
from acarla.record.writer import TraceWriter  # noqa: E402
from acarla.sim import groundtruth  # noqa: E402
from acarla.sim.attach import apply_offset, rig_to_actor_offset  # noqa: E402
from acarla.sim.client import SimSession  # noqa: E402
from acarla.sim.coords import carla_rotation_to_rig_rotation, carla_to_rig_xyz  # noqa: E402
from acarla.sim.rig import load_rig_config, load_rig_intrinsics, load_rig_vehicle  # noqa: E402
from acarla.types import (  # noqa: E402
    N_EGO_WAYPOINTS,
    N_HISTORY_FRAMES,
    SAMPLE_DT,
    BoundingBox,
    ControlCommand,
    PlanResult,
    RunHeader,
    TraceFrame,
)

sys.path.insert(0, str(ROOT / "scripts"))
from run_open_loop import DEFAULT_VEHICLE_BLUEPRINT, VEHICLE_MESH_SHIFT_X  # noqa: E402

DEFAULT_RIG_PATH = ROOT / "configs" / "rig_alpamayo.yaml"
GROUND_TRUTH_RADIUS_M = 60.0
_EGO_STRIDE = 2  # trace ticks (0.05 s) per model ego sample (0.1 s), as in build_model_inputs


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--map", default="Town10HD_Opt")
    p.add_argument("--ticks", type=int, default=1200, help="world ticks at 0.05 s (1200 = 60 s)")
    p.add_argument(
        "--replan-interval",
        type=float,
        default=1.0,
        help="simulated seconds between Alpamayo queries (the controller's max plan age "
        "is set to this + 0.25 s)",
    )
    p.add_argument("--traffic", type=int, default=20)
    p.add_argument("--rig", type=Path, default=DEFAULT_RIG_PATH)
    p.add_argument("--vehicle", default=DEFAULT_VEHICLE_BLUEPRINT)
    p.add_argument(
        "--rig-shift-x",
        type=float,
        default=None,
        help="rigid +x shift of the whole rig (default: VEHICLE_MESH_SHIFT_X of the vehicle)",
    )
    p.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="private HF dataset repo used as the queue, or dir:<path> for a local directory queue "
        "(smoke tests with scripts/fake_loop_worker.py)",
    )
    p.add_argument(
        "--run-id", default=None, help="queue run id (default: loop-<utc timestamp>-seed<seed>)"
    )
    p.add_argument(
        "--codec",
        choices=("jpg", "png"),
        default="jpg",
        help="frame encoding on the wire (png = bit-exact batch tensors, ~8x larger jobs)",
    )
    p.add_argument("--poll", type=float, default=3.0, help="seconds between plan polls")
    p.add_argument(
        "--plan-wait-max",
        type=float,
        default=0.0,
        help="abort if one plan takes longer than this many wall seconds (0 = wait forever, "
        "e.g. across a Kaggle session restart)",
    )
    p.add_argument(
        "--warmup-s",
        type=float,
        default=2.0,
        help="simulated seconds before the first query (>= 1.5 s for 16 ego samples)",
    )
    p.add_argument(
        "--mark-others-done",
        action="store_true",
        help="mark every other run in the queue DONE first so the worker only serves this one",
    )
    p.add_argument(
        "--stop-worker-at-end",
        action="store_true",
        help="write worker/STOP after the run (worker exits, saves GPU quota)",
    )
    p.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="format of the per-tick camera images written into the trace",
    )
    p.add_argument(
        "--no-aeb",
        action="store_true",
        help="disable the ground-truth emergency brake (default on; interventions are counted "
        "and stamped into the trace, see acarla.control.aeb)",
    )
    p.add_argument(
        "--stop-on-collision",
        action="store_true",
        help="end the run at the first collision (now the default)",
    )
    p.add_argument(
        "--continue-on-collision",
        action="store_true",
        help="diagnostic only: keep running after contact; the evaluation still fails",
    )
    p.add_argument(
        "--no-supervisor",
        action="store_true",
        help="ablation only: disable the simulator lane/static-obstacle safety check",
    )
    p.add_argument("--purge-actors", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument(
        "--allow-fake-worker",
        action="store_true",
        help="local transport smoke only; never reported as Alpamayo",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args(argv)


def _config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(
        {
            "planner": "alpamayo-1.5-10b-nf4-t4x2-remote",
            "controller": asdict(
                ControllerConfig(
                    replan_interval_s=args.replan_interval,
                    max_plan_age_s=args.replan_interval + 0.25,
                )
            ),
            "safety_revision": "environment-footprint-v1",
            "aeb": not args.no_aeb,
            "supervisor": not args.no_supervisor,
            "rig_shift_x": (
                args.rig_shift_x
                if args.rig_shift_x is not None
                else VEHICLE_MESH_SHIFT_X.get(args.vehicle, 0.0)
            ),
            "replan_interval_s": args.replan_interval,
            "codec": args.codec,
            "rig": str(args.rig.name),
            "rig_sha256": hashlib.sha256(args.rig.read_bytes()).hexdigest(),
            "vehicle": args.vehicle,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _speed(v: carla.Vector3D) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _wait_for_plan(
    q: LoopQueue, run_id: str, seq: int, args: argparse.Namespace
) -> tuple[PlanResult | None, dict, float]:
    """Block (CARLA stays paused) until the worker answered job `seq`."""
    t0 = time.monotonic()
    last_report = 0.0
    while True:
        text = q.get_plan(run_id, seq)
        if text is not None:
            doc = json.loads(text)
            if doc.get("run_id") != run_id or doc.get("seq") != seq:
                raise ValueError("worker response belongs to a different run/job")
            if not doc.get("finite", False):  # worker-side error answer, see kaggle_loop_worker
                return None, doc, time.monotonic() - t0
            plan, doc = decode_plan(text, expect_run_id=run_id, expect_seq=seq)
            return plan, doc, time.monotonic() - t0
        waited = time.monotonic() - t0
        if args.plan_wait_max > 0 and waited > args.plan_wait_max:
            raise TimeoutError(f"no plan for job {seq} after {waited:.0f} s")
        if waited - last_report >= 60.0:
            last_report = waited
            hb = q.heartbeat()
            if hb is None:
                status = (
                    "no worker heartbeat yet -- start notebooks/kaggle_loop_worker.ipynb on Kaggle"
                )
            else:
                age = time.time() - float(hb.get("ts", 0))
                status = (
                    f"worker state={hb.get('state')} session={hb.get('session')} "
                    f"done={hb.get('n_done')} heartbeat {age:.0f} s ago"
                    + (
                        "  [STALE -- kernel probably ended; restart it in the browser]"
                        if age > 420
                        else ""
                    )
                )
            print(f"  waiting for plan {seq}: {waited:.0f} s | {status}", flush=True)
        time.sleep(args.poll)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ticks <= 0:
        raise ValueError("--ticks must be positive")
    if not math.isfinite(args.replan_interval) or not 0.05 <= args.replan_interval <= 2.0:
        raise ValueError("--replan-interval must be between 0.05 and 2.0 seconds")
    if args.poll <= 0 or not math.isfinite(args.poll):
        raise ValueError("--poll must be finite and positive")
    if not math.isfinite(args.warmup_s) or args.warmup_s < (N_EGO_WAYPOINTS - 1) * SAMPLE_DT:
        raise ValueError("--warmup-s must cover 16 ego samples (>= 1.5 s)")
    if not math.isfinite(args.plan_wait_max) or args.plan_wait_max < 0:
        raise ValueError("--plan-wait-max must be finite and nonnegative")
    dt_s = 0.05
    rng = random.Random(args.seed)
    run_id = args.run_id or f"loop-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}-seed{args.seed}"
    config_hash = _config_hash(args)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"output directory is not empty; choose a new --out: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    # -- queue ---------------------------------------------------------------
    q = open_queue(args.repo)
    print(f"queue: {args.repo} as {q.whoami()}, run {run_id}")
    q.ensure_repo()
    if hasattr(q, "publish_code"):
        q.publish_code(SRC / "acarla")  # worker refreshes codec/transport from here at start
    if args.mark_others_done:
        for other in q.list_runs():
            if other != run_id and not q.run_is_done(other):
                q.finish_run(other, {"reason": "superseded", "by": run_id})
                print(f"  marked {other} DONE")
    hb = q.heartbeat()
    if hb is None:
        print("worker: no heartbeat yet (jobs will queue until the Kaggle notebook is running)")
    else:
        print(
            f"worker: state={hb.get('state')} session={hb.get('session')} "
            f"heartbeat {time.time() - float(hb.get('ts', 0)):.0f} s ago"
        )

    # -- rig -----------------------------------------------------------------
    specs = load_rig_config(args.rig)
    ftheta_models = load_rig_intrinsics(args.rig)
    camera_names = [s.name for s in specs]
    remaps = {}
    for s in specs:
        pinhole = PinholeModel(width=s.width, height=s.height, hfov_deg=s.fov)
        remaps[s.name] = (build_remap(ftheta_models[s.name], pinhole), pinhole)

    collision_frames: list[int] = []
    plans_log: list[dict] = []
    summary: dict = {"status": "aborted", "run_id": run_id}
    started_queue_run = False
    try:
        with SimSession(
            host=args.host,
            port=args.port,
            map_name=args.map,
            timeout_s=args.timeout,
            fixed_delta_seconds=dt_s,
            tm_port=args.tm_port,
        ) as session:
            session.configure_sync(args.seed, args.seed)
            if args.purge_actors:
                print(f"purged {session.purge_actors()} pre-existing actor(s)")
            ego = session.spawn_ego(rng, blueprint_filter=args.vehicle, autopilot=False)
            session.tick()
            traffic = session.spawn_traffic(
                args.traffic,
                rng,
                exclude_spawn_point=ego.get_transform(),
                near_location=ego.get_transform().location,
                max_distance_m=GROUND_TRUTH_RADIUS_M,
            )
            print(f"spawned {len(traffic)}/{args.traffic} traffic vehicle(s)")
            session.spawn_collision_sensor(ego, lambda e: collision_frames.append(int(e.frame)))
            scene = groundtruth.SceneCache(session.world)
            print(f"cached {len(scene.objects)} static environment objects")

            # rig attachment exactly as in run_open_loop.py (axle offset + mesh shift)
            bbox = ego.bounding_box
            ego_box = BoundingBox(
                np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z], np.float32),
                np.array([bbox.location.x, bbox.location.y, bbox.location.z], np.float32),
            )
            aeb_config = AebConfig(
                ego_front_m=float(bbox.extent.x + bbox.location.x),
                ego_rear_m=float(bbox.extent.x - bbox.location.x),
                corridor_half_width_m=float(bbox.extent.y + 0.3),
            )

            is_drivable = groundtruth.make_drivable_check(scene.map)

            vehicle_cfg = load_rig_vehicle(args.rig)
            offset = rig_to_actor_offset(
                float(vehicle_cfg["rear_axle_to_bbox_center"]),
                bbox.extent.z,
                (bbox.location.x, bbox.location.y, bbox.location.z),
            )
            shift_x = (
                args.rig_shift_x
                if args.rig_shift_x is not None
                else VEHICLE_MESH_SHIFT_X.get(args.vehicle, 0.0)
            )
            attached = [
                apply_offset(apply_offset(s, offset), np.array([shift_x, 0.0, 0.0])) for s in specs
            ]
            (args.out / "rig_used.json").write_text(
                json.dumps(
                    {
                        "rig_path": str(args.rig),
                        "vehicle_blueprint": args.vehicle,
                        "applied_offset": [float(v) for v in offset],
                        "vehicle_mesh_shift_x": float(shift_x),
                        "cameras": [
                            {
                                "name": s.name,
                                "x": s.x,
                                "y": s.y,
                                "z": s.z,
                                "roll": s.roll,
                                "pitch": s.pitch,
                                "yaw": s.yaw,
                                "fov": s.fov,
                                "width": s.width,
                                "height": s.height,
                            }
                            for s in attached
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            session.spawn_cameras(ego, attached)

            header = RunHeader(
                carla_version=session.client.get_client_version(),
                map_name=session.world.get_map().name,
                seed_world=args.seed,
                seed_traffic_manager=args.seed,
                model_config_name=(
                    "transport-smoke"
                    if args.allow_fake_worker
                    else "alpamayo-1.5-10b-nf4-t4x2-remote-slowmotion-closed-loop"
                ),
                model_config_hash=config_hash,
                git_sha=_git_sha(),
                run_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                cameras=camera_names,
                fixed_delta_seconds=dt_s,
                environment_objects=scene.objects,
            )
            q.start_run(
                run_id,
                {
                    "map": header.map_name,
                    "seed": args.seed,
                    "ticks": args.ticks,
                    "replan_interval_s": args.replan_interval,
                    "codec": args.codec,
                    "git_sha": header.git_sha,
                    "host": "local-carla",
                },
            )
            started_queue_run = True

            controller = TrajectoryController(
                ControllerConfig(
                    replan_interval_s=args.replan_interval,
                    max_plan_age_s=args.replan_interval + 0.25,
                )
            )
            current_plan: PlanResult | None = None
            last_plan_time: float | None = None
            seq = 0
            ego_hist: deque[tuple[float, object]] = deque(
                maxlen=(N_EGO_WAYPOINTS - 1) * _EGO_STRIDE + 1
            )
            img_hist: dict[str, deque] = {c: deque(maxlen=N_HISTORY_FRAMES) for c in camera_names}
            warmup_ticks = int(round(args.warmup_s / dt_s))
            distance_m = 0.0
            previous_xyz = None
            speeds: list[float] = []
            offroad_ticks = 0
            footprint_offroad_ticks = 0
            unsafe_plans = 0
            fault_ticks = 0
            aeb_ticks = 0
            aeb_events: list[dict] = []
            supervisor_ticks = 0
            supervisor_events: list[dict] = []
            worker_errors = 0
            n_collisions_seen = 0
            ticks_done = 0
            observed_warmup_ticks = 0
            t_run0 = time.monotonic()

            with TraceWriter(args.out, header, image_ext=args.image_format) as writer:
                for local_frame in range(args.ticks):
                    carla_frame = session.tick()
                    sim_time = round(local_frame * dt_s, 6)
                    images = session.collect_images(carla_frame, camera_names)
                    if images and len(images) < len(camera_names):
                        raise RuntimeError(
                            f"tick {local_frame}: only {sorted(images)} delivered an image"
                        )

                    image_paths: dict[str, list[str]] = {}
                    for cam, image in images.items():
                        rgb = np.ascontiguousarray(
                            np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                                image.height, image.width, 4
                            )[:, :, :3][:, :, ::-1]
                        )
                        img_hist[cam].append((sim_time, rgb))
                        path = writer.image_path(cam, local_frame)
                        if args.image_format == "jpg":
                            Image.fromarray(rgb).save(str(path), quality=92)
                        else:
                            image.save_to_disk(str(path))
                        image_paths[cam] = [writer.relative_image_path(cam, local_frame)]

                    ego_transform = ego.get_transform()
                    ego_pose = groundtruth.transform_to_pose(ego_transform)
                    ego_hist.append((sim_time, ego_pose))

                    due = (
                        last_plan_time is None
                        or sim_time - last_plan_time >= args.replan_interval - 1e-9
                    )
                    ready = (
                        bool(images)
                        and local_frame >= warmup_ticks
                        and len(ego_hist) == ego_hist.maxlen
                        and all(len(h) == N_HISTORY_FRAMES for h in img_hist.values())
                    )
                    if due and ready:
                        t_step0 = time.monotonic()
                        frames_hwc, image_ts = {}, {}
                        for cam in camera_names:
                            (map_x, map_y), _ = remaps[cam]
                            frames_hwc[cam] = np.stack(
                                [apply_remap(rgb, map_x, map_y) for _, rgb in img_hist[cam]]
                            )
                            image_ts[cam] = [round(t, 6) for t, _ in img_hist[cam]]
                        hist = list(ego_hist)[::_EGO_STRIDE]  # oldest first, t0 last
                        assert len(hist) == N_EGO_WAYPOINTS and hist[-1][0] == sim_time
                        world_xyz = np.array(
                            [carla_to_rig_xyz(p.translation.astype(np.float64)) for _, p in hist]
                        )
                        world_quat = np.array(
                            [
                                Rotation.from_matrix(
                                    carla_rotation_to_rig_rotation(p.rotation.astype(np.float64))
                                ).as_quat()
                                for _, p in hist
                            ]
                        )
                        packet = build_sensor_packet(
                            frame_id=local_frame,
                            sim_time=sim_time,
                            frames_hwc=frames_hwc,
                            image_timestamps=image_ts,
                            world_xyz=world_xyz,
                            world_quat_xyzw=world_quat,
                            ego_timestamps=[round(t, 6) for t, _ in hist],
                        )
                        model_inputs = sensor_packet_to_model_inputs(packet)
                        payload = encode_job(
                            model_inputs,
                            run_id=run_id,
                            seq=seq,
                            frame_id=local_frame,
                            sim_time=sim_time,
                            codec=args.codec,
                        )
                        t_enc = time.monotonic() - t_step0
                        q.put_job(run_id, seq, payload)
                        t_up = time.monotonic() - t_step0 - t_enc
                        print(
                            f"[t={sim_time:6.2f}s] job {seq}: {len(payload) / 1e6:.1f} MB, "
                            f"encode {t_enc:.1f}s, upload {t_up:.1f}s, "
                            f"v={_speed(ego.get_velocity()):.1f} m/s -- waiting",
                            flush=True,
                        )
                        plan, doc, t_wait = _wait_for_plan(q, run_id, seq, args)
                        if plan is None:
                            controller.invalidate_plan("worker error")
                            current_plan = None
                            worker_errors += 1
                            print(
                                f"           job {seq}: WORKER ERROR {doc.get('reasoning')!r} "
                                "-- controller brakes, "
                                f"retrying next interval ({worker_errors}/3)",
                                flush=True,
                            )
                            last_plan_time = sim_time
                            seq += 1
                            if worker_errors >= 3:
                                raise RuntimeError(
                                    "worker failed three jobs in a row; see Kaggle log"
                                )
                        else:
                            is_fake = doc.get("model_config_hash", "").startswith("fake-") or str(
                                doc.get("worker_session", "")
                            ).startswith("fake-")
                            if is_fake and not args.allow_fake_worker:
                                raise ValueError("fake worker requires --allow-fake-worker")
                            if plan.frame_id != local_frame:
                                raise ValueError("worker plan is anchored to the wrong frame")
                            worker_errors = 0
                            safety = controller.set_plan(plan, ego_pose, sim_time)
                            if not safety.safe:
                                unsafe_plans += 1
                            current_plan, last_plan_time = plan, sim_time
                            round_trip = time.monotonic() - t_step0
                            plans_log.append(
                                {
                                    "seq": seq,
                                    "frame_id": local_frame,
                                    "sim_time": sim_time,
                                    "waypoints_xyz": plan.waypoints_xyz.tolist(),
                                    "waypoints_rot": plan.waypoints_rot.tolist(),
                                    "reasoning": plan.reasoning,
                                    "inference_ms": plan.inference_ms,
                                    "model_config_hash": plan.model_config_hash,
                                    "finite": True,
                                    "end_xyz": plan.waypoints_xyz[-1].tolist(),
                                    "path_length_m": float(
                                        np.linalg.norm(
                                            np.diff(plan.waypoints_xyz[:, :2], axis=0), axis=1
                                        ).sum()
                                    ),
                                    "safety": {
                                        "safe": safety.safe,
                                        "reason": safety.reason,
                                        "max_abs_curvature_inv_m": safety.max_abs_curvature_inv_m,
                                        "max_abs_acceleration_mps2": (
                                            safety.max_abs_acceleration_mps2
                                        ),
                                        "max_speed_mps": safety.max_speed_mps,
                                    },
                                    "worker_session": doc.get("worker_session"),
                                    "job_bytes": len(payload),
                                    "wall": {
                                        "encode_s": t_enc,
                                        "upload_s": t_up,
                                        "wait_s": t_wait,
                                        "round_trip_s": round_trip,
                                    },
                                    "ego_pose_world": ego_pose.to_json_dict(),
                                }
                            )
                            (args.out / "plans.json").write_text(
                                json.dumps(plans_log), encoding="utf-8"
                            )
                            end = plan.waypoints_xyz[-1]
                            print(
                                f"           plan {seq}: end=({end[0]:.1f},{end[1]:.1f}) m, "
                                f"inference {plan.inference_ms / 1000:.1f}s, "
                                f"round trip {round_trip:.0f}s, safe={safety.safe} "
                                f"({safety.reason}) | {plan.reasoning!r}",
                                flush=True,
                            )
                            seq += 1

                    speed = _speed(ego.get_velocity())
                    speeds.append(speed)
                    decision = controller.step(
                        ego_pose, current_speed_mps=speed, sim_time_s=sim_time, dt=dt_s
                    )
                    warming_up = seq == 0
                    observed_warmup_ticks += int(warming_up)
                    if not decision.safe and not warming_up:
                        fault_ticks += 1
                    command = decision.command
                    actors_now = groundtruth.actor_states(
                        session.world, ego.id, GROUND_TRUTH_RADIUS_M
                    )
                    static_now = scene.nearby(ego_pose.translation)
                    aeb_active = False
                    if not args.no_aeb:
                        aeb = check_forward_corridor(ego_pose, speed, actors_now, aeb_config)
                        if aeb.brake:
                            aeb_active = True
                            command = apply_aeb(command, aeb)
                            aeb_ticks += 1
                            if not aeb_events or aeb_events[-1]["end_tick"] < local_frame - 1:
                                aeb_events.append(
                                    {
                                        "start_tick": local_frame,
                                        "end_tick": local_frame,
                                        "sim_time": sim_time,
                                        "gap_m": aeb.gap_m,
                                        "reason": aeb.reason,
                                        "speed_mps": speed,
                                    }
                                )
                                print(
                                    f"           AEB at t={sim_time:.2f}s: {aeb.reason} "
                                    f"(v={speed:.1f} m/s)",
                                    flush=True,
                                )
                            else:
                                aeb_events[-1]["end_tick"] = local_frame
                    supervised = SupervisorDecision()
                    if not args.no_supervisor and not warming_up:
                        supervised = check_environment(
                            ego_pose,
                            speed,
                            command,
                            controller.world_waypoints,
                            actors_now + static_now,
                            is_drivable,
                            ego_box,
                        )
                        if supervised.brake:
                            command = ControlCommand(command.steer, 0.0, 1.0)
                            supervisor_ticks += 1
                            if (
                                not supervisor_events
                                or supervisor_events[-1]["end_tick"] < local_frame - 1
                                or supervisor_events[-1]["reason"] != supervised.reason
                            ):
                                supervisor_events.append(
                                    {
                                        "start_tick": local_frame,
                                        "end_tick": local_frame,
                                        "sim_time": sim_time,
                                        **asdict(supervised),
                                    }
                                )
                            else:
                                supervisor_events[-1]["end_tick"] = local_frame
                    if aeb_active or supervised.brake:
                        controller.reset_after_override()
                    ego.apply_control(
                        carla.VehicleControl(
                            steer=command.steer, throttle=command.throttle, brake=command.brake
                        )
                    )

                    loc = ego_transform.location
                    xyz = np.array([loc.x, loc.y, loc.z])
                    if previous_xyz is not None:
                        distance_m += float(np.linalg.norm(xyz - previous_xyz))
                    previous_xyz = xyz
                    offroad = (
                        scene.map.get_waypoint(
                            loc, project_to_road=False, lane_type=carla.LaneType.Driving
                        )
                        is None
                    )
                    if offroad:
                        offroad_ticks += 1
                    footprint_offroad = not footprint_is_drivable(ego_pose, ego_box, is_drivable)
                    footprint_offroad_ticks += int(footprint_offroad)
                    writer.write_frame(
                        TraceFrame(
                            frame_id=local_frame,
                            sim_time=sim_time,
                            ego_pose_world=ego_pose,
                            actors=actors_now,
                            lanes=scene.lanes(loc, GROUND_TRUTH_RADIUS_M),
                            traffic_lights=groundtruth.traffic_light_states(
                                session.world, loc, GROUND_TRUTH_RADIUS_M
                            ),
                            plan=current_plan,
                            control=command,
                            image_paths=image_paths,
                            diagnostics={
                                "speed_mps": speed,
                                "target_speed_mps": decision.target_speed_mps,
                                "warmup": warming_up,
                                "controller_safe": decision.safe,
                                "controller_reason": decision.reason,
                                "aeb": aeb_active,
                                "supervisor": asdict(supervised),
                                "offroad": offroad,
                                "footprint_offroad": footprint_offroad,
                                "collision": carla_frame in collision_frames,
                                "carla_frame": carla_frame,
                                "ground_truth_assistance": not args.no_aeb
                                or not args.no_supervisor,
                            },
                        )
                    )
                    ticks_done += 1
                    summary.update(
                        ticks_completed=ticks_done,
                        ticks_requested=args.ticks,
                        simulated_seconds=ticks_done * dt_s,
                        n_plans=len(plans_log),
                        n_collision_frames=len(set(collision_frames)),
                        offroad_ticks=offroad_ticks,
                        footprint_offroad_ticks=footprint_offroad_ticks,
                        distance_m=distance_m,
                        aeb_ticks=aeb_ticks,
                        supervisor_ticks=supervisor_ticks,
                        fault_ticks=fault_ticks,
                    )
                    if len(collision_frames) > n_collisions_seen:
                        n_collisions_seen = len(collision_frames)
                        print(
                            f"           COLLISION at tick {local_frame} "
                            f"(t={sim_time:.2f}s, v={speed:.1f} m/s)",
                            flush=True,
                        )
                        if args.stop_on_collision or not args.continue_on_collision:
                            break

        wall = time.monotonic() - t_run0
        summary = {
            "status": "closed_loop_completed"
            if ticks_done == args.ticks
            else "closed_loop_stopped",
            "alpamayo_used": bool(plans_log)
            and all(
                not p["model_config_hash"].startswith("fake-")
                and not str(p["worker_session"]).startswith("fake-")
                for p in plans_log
            ),
            "closed_loop": True,
            "real_time": False,
            "run_id": run_id,
            "queue_repo": args.repo,
            "map": header.map_name,
            "seed": args.seed,
            "ticks_requested": args.ticks,
            "ticks_completed": ticks_done,
            "simulated_seconds": ticks_done * dt_s,
            "wall_seconds": wall,
            "slowdown_factor": wall / max(ticks_done * dt_s, 1e-9),
            "replan_interval_s": args.replan_interval,
            "n_plans": len(plans_log),
            "unsafe_plans": unsafe_plans,
            "fault_ticks": fault_ticks,
            "aeb_enabled": not args.no_aeb,
            "aeb_ticks": aeb_ticks,
            "aeb_events": aeb_events,
            "supervisor_enabled": not args.no_supervisor,
            "supervisor_ticks": supervisor_ticks,
            "supervisor_events": supervisor_events,
            "static_objects_recorded": len(scene.objects),
            "warmup_ticks": observed_warmup_ticks,
            "collision_frames": sorted(set(collision_frames)),
            "n_collision_frames": len(set(collision_frames)),
            "offroad_ticks": offroad_ticks,
            "footprint_offroad_ticks": footprint_offroad_ticks,
            "git_sha": header.git_sha,
            "distance_m": distance_m,
            "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed_mps": float(np.max(speeds)) if speeds else 0.0,
            "mean_inference_s": float(np.mean([p["inference_ms"] for p in plans_log]) / 1000)
            if plans_log
            else None,
            "mean_round_trip_s": float(np.mean([p["wall"]["round_trip_s"] for p in plans_log]))
            if plans_log
            else None,
            "worker_sessions": sorted(
                {p["worker_session"] for p in plans_log if p.get("worker_session")}
            ),
            "model_config_hash": config_hash,
        }
        summary["evaluation"] = evaluate_run(summary)
        return 0 if summary["evaluation"]["safe_completion"] else 2
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["evaluation"] = evaluate_run(summary)
        raise
    finally:
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "worker_sessions"}, indent=2))
        if started_queue_run:
            try:
                q.finish_run(
                    run_id, {k: summary.get(k) for k in ("status", "n_plans", "simulated_seconds")}
                )
                if args.stop_worker_at_end:
                    q.request_stop()
            except Exception as exc:  # never mask the primary error
                print(f"warning: could not finish queue run: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
