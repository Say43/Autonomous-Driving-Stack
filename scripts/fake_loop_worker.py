#!/usr/bin/env python3
"""Stand-in for the Kaggle worker: answers every queued job with a straight
constant-speed plan. Exercises the queue protocol, the job codec and the
CARLA side of `run_closed_loop_alpamayo.py` end to end without a GPU, a
Hugging Face login or Kaggle. Not Alpamayo -- the plan says so.

Usage (two terminals, or run this one in the background first):
    python scripts/fake_loop_worker.py --repo dir:results/_fake_queue --speed 6
    python scripts/run_closed_loop_alpamayo.py --repo dir:results/_fake_queue --out results/loop_smoke --seed 1 --ticks 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.loop.codec import decode_job, encode_plan  # noqa: E402
from acarla.loop.transport import open_queue  # noqa: E402
from acarla.types import N_PLAN_WAYPOINTS, SAMPLE_DT  # noqa: E402


def straight_plan(speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros((N_PLAN_WAYPOINTS, 3))
    xyz[:, 0] = speed_mps * SAMPLE_DT * np.arange(1, N_PLAN_WAYPOINTS + 1)
    rot = np.tile(np.eye(3), (N_PLAN_WAYPOINTS, 1, 1))
    return xyz, rot


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="queue id, e.g. dir:results/_fake_queue")
    p.add_argument("--speed", type=float, default=6.0, help="m/s of the straight plan")
    p.add_argument("--latency", type=float, default=1.0, help="fake inference seconds per job")
    p.add_argument("--max-jobs", type=int, default=0, help="exit after this many jobs (0 = until STOP)")
    p.add_argument("--poll", type=float, default=0.5)
    args = p.parse_args(argv)

    q = open_queue(args.repo)
    q.ensure_repo()
    session = f"fake-{time.strftime('%H%M%S')}"
    q.write_heartbeat({"session": session, "state": "idle", "model_loaded": True, "n_done": 0})
    n_done = 0
    done_runs: set[str] = set()
    while True:
        if q.stop_requested():
            q.clear_stop()
            break
        work = None
        for run_id in reversed(q.list_runs()):
            if run_id in done_runs:
                continue
            if q.run_is_done(run_id):
                done_runs.add(run_id)
                continue
            pending = q.pending_jobs(run_id)
            if pending:
                work = (run_id, pending[0])
                break
        if work is None:
            time.sleep(args.poll)
            continue
        run_id, seq = work
        job = decode_job(q.get_job(run_id, seq))
        assert job["image_frames"].shape[1:3] == (4, 3), job["image_frames"].shape
        time.sleep(args.latency)
        xyz, rot = straight_plan(args.speed)
        q.put_plan(run_id, seq, encode_plan(
            run_id=run_id, seq=seq, frame_id=job["meta"]["frame_id"], sim_time=job["meta"]["sim_time"],
            waypoints_xyz=xyz, waypoints_rot=rot,
            reasoning=f"FAKE WORKER: straight at {args.speed} m/s; not Alpamayo output",
            inference_ms=args.latency * 1000.0, model_config_hash="fake-straight", worker_session=session,
        ))
        n_done += 1
        print(f"{run_id} job {seq}: frame {job['meta']['frame_id']} t={job['meta']['sim_time']:.2f}s "
              f"-> straight plan ({n_done})", flush=True)
        q.write_heartbeat({"session": session, "state": "idle", "model_loaded": True, "n_done": n_done})
        if args.max_jobs and n_done >= args.max_jobs:
            break
    q.write_heartbeat({"session": session, "state": "exited", "n_done": n_done})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
