"""Queue protocol of the slow-motion closed loop, on the directory backend."""

import json

import numpy as np

from acarla.loop.codec import encode_job
from acarla.loop.transport import LocalDirQueue, open_queue


def _job(seq):
    mi = {
        "image_frames": np.zeros((4, 4, 3, 8, 12), np.uint8),
        "camera_indices": np.array([0, 1, 2, 6]),
        "ego_history_xyz": np.zeros((1, 1, 16, 3), np.float32),
        "ego_history_rot": np.tile(np.eye(3, dtype=np.float32), (1, 1, 16, 1, 1)),
    }
    return encode_job(mi, run_id="r", seq=seq, frame_id=seq * 20, sim_time=seq * 1.0, codec="png")


def test_open_queue_dir_prefix(tmp_path):
    q = open_queue(f"dir:{tmp_path / 'q'}")
    assert isinstance(q, LocalDirQueue)
    assert q.whoami() == "local"


def test_producer_consumer_protocol(tmp_path):
    q = LocalDirQueue(tmp_path)
    assert q.list_runs() == [] and q.heartbeat() is None
    q.start_run("r", {"map": "Town10HD_Opt"})
    assert q.list_runs() == ["r"] and not q.run_is_done("r")
    assert q.pending_jobs("r") == []

    q.put_job("r", 0, _job(0))
    q.put_job("r", 1, _job(1))
    assert q.pending_jobs("r") == [0, 1]
    assert q.get_plan("r", 0) is None

    q.put_plan("r", 0, json.dumps({"seq": 0}))
    assert q.pending_jobs("r") == [1]
    assert json.loads(q.get_plan("r", 0)) == {"seq": 0}
    assert q.get_job("r", 1) == _job(1)

    q.write_heartbeat({"state": "idle"})
    hb = q.heartbeat()
    assert hb["state"] == "idle" and "ts" in hb

    assert not q.stop_requested()
    q.request_stop()
    assert q.stop_requested()
    q.clear_stop()
    assert not q.stop_requested()

    q.finish_run("r", {"n_plans": 1})
    assert q.run_is_done("r")


def test_partial_files_are_invisible(tmp_path):
    q = LocalDirQueue(tmp_path)
    q.start_run("r", {})
    (tmp_path / "runs/r/jobs").mkdir(parents=True)
    (tmp_path / "runs/r/jobs/00007.npz.part").write_bytes(b"x")
    assert q.pending_jobs("r") == []


def test_seq_parsing_ignores_foreign_files():
    assert LocalDirQueue._seqs(["runs/r/jobs/00003.npz", "runs/r/jobs/readme.txt", "x/00010.json"]) == [3, 10]
