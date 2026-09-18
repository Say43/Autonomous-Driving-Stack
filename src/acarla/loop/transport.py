"""Job/plan queue on a private Hugging Face *dataset* repository.

Why HF and not a socket: the local machine has no public address, the Kaggle
kernel has outbound internet plus the Hugging Face secret it already needs
for the gated backbone weights. Every transfer is a small commit; both sides
poll. Layout inside the repo::

    runs/<run_id>/run.json            written by the local loop at start
    runs/<run_id>/jobs/<seq:05d>.npz   one job per replan (see codec.encode_job)
    runs/<run_id>/plans/<seq:05d>.json one answer per job (codec.encode_plan)
    runs/<run_id>/DONE                local loop finished (worker skips the run)
    worker/heartbeat.json             worker state, rewritten every minute
    worker/STOP                       ask the worker to exit after the current job

Rate limit: the Hub allows 128 commits per hour *per repository*. One replan
costs one job commit and one plan commit, so a 1 s replan loop (~130 steps/h)
would exhaust a single repo twice over. Jobs and plans are therefore sharded
over separate repos (`<base>-j0`, `<base>-j1`, `<base>-p0`, `<base>-p1`, seq
mod 2), while `run.json`, `DONE`, `heartbeat.json`, `STOP` stay in the base
repo. A 429 is waited out (up to ~70 min) rather than treated as fatal.

The token is never handled here: `huggingface_hub` reads it from the
environment (`HF_TOKEN`) or the local login (`hf auth login`).
"""

from __future__ import annotations

import io
import json
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

try:  # optional: the directory backend and the unit tests need no hub client
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover
    HfApi = hf_hub_download = None

    class HfHubHTTPError(Exception):
        pass

DEFAULT_REPO = "acarla-loop"  # no namespace: resolved to <hf username>/acarla-loop at runtime
N_SHARDS = 2
RATE_LIMIT_WAIT_S = 70 * 60
_SEQ_RE = re.compile(r"(\d{5})\.(npz|json)$")


def _status(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _is_not_found(exc: Exception) -> bool:
    if _status(exc) == 404:
        return True
    text = str(exc).lower()
    return "404" in text or "not found" in text


def _retry(fn: Callable[[], Any], *, attempts: int = 6, what: str = "hf call") -> Any:
    """Retry transient HTTP failures (5xx/network) with jittered backoff; wait
    out a 429 commit-rate limit (128/h/repo) for up to RATE_LIMIT_WAIT_S."""
    delay = 2.0
    last: Exception | None = None
    attempt = 0
    rate_limited_since: float | None = None
    while True:
        attempt += 1
        try:
            return fn()
        except HfHubHTTPError as exc:
            status = _status(exc)
            if status == 429:
                rate_limited_since = rate_limited_since or time.time()
                if time.time() - rate_limited_since > RATE_LIMIT_WAIT_S:
                    raise RuntimeError(
                        f"{what}: still rate limited after {RATE_LIMIT_WAIT_S // 60} min"
                    ) from exc
                print(f"  [{what}] HF rate limit (429) -- waiting 60 s "
                      f"({(time.time() - rate_limited_since) / 60:.0f} min so far)", flush=True)
                time.sleep(60.0)
                attempt -= 1  # rate limiting does not count against the attempt budget
                continue
            if status is not None and status < 500 and status != 408:
                raise
            last = exc
        except (ConnectionError, TimeoutError, OSError) as exc:
            last = exc
        if attempt >= attempts:
            raise RuntimeError(f"{what} failed after {attempts} attempts: {last}") from last
        time.sleep(delay + random.uniform(0.0, 1.0))
        delay = min(delay * 2.0, 60.0)


class LoopQueue:
    def __init__(self, repo_id: str = DEFAULT_REPO, cache_dir: str | Path | None = None) -> None:
        if HfApi is None:
            raise ImportError("huggingface_hub is required for the Hugging Face queue backend")
        self.api = HfApi()
        if "/" not in repo_id:  # HF user (say43) != Kaggle user (says43): never hard-code the namespace
            repo_id = f"{self.api.whoami()['name']}/{repo_id}"
        self.repo_id = repo_id
        self.job_repos = [f"{repo_id}-j{i}" for i in range(N_SHARDS)]
        self.plan_repos = [f"{repo_id}-p{i}" for i in range(N_SHARDS)]
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="acarla_loop_"))
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- setup -----------------------------------------------------------
    def ensure_repo(self) -> None:
        for repo in [self.repo_id, *self.job_repos, *self.plan_repos]:
            _retry(
                lambda repo=repo: self.api.create_repo(
                    repo, repo_type="dataset", private=True, exist_ok=True
                ),
                what=f"create_repo {repo}",
            )

    def whoami(self) -> str:
        return str(_retry(lambda: self.api.whoami(), what="whoami").get("name"))

    # -- primitives ------------------------------------------------------
    def _upload(self, path_in_repo: str, data: bytes, message: str, repo: str | None = None) -> None:
        repo = repo or self.repo_id
        _retry(
            lambda: self.api.upload_file(
                path_or_fileobj=io.BytesIO(data),
                path_in_repo=path_in_repo,
                repo_id=repo,
                repo_type="dataset",
                commit_message=message,
            ),
            what=f"upload {repo}:{path_in_repo}",
        )

    def _download(self, path_in_repo: str, *, mutable: bool, repo: str | None = None) -> bytes | None:
        """Fetch a file's bytes; None if it does not exist. Mutable files
        (heartbeat, run.json) bypass the content-addressed cache."""
        repo = repo or self.repo_id

        def go() -> bytes | None:
            try:
                local = hf_hub_download(
                    repo,
                    path_in_repo,
                    repo_type="dataset",
                    cache_dir=str(self.cache_dir),
                    force_download=mutable,
                )
            except Exception as exc:
                if _is_not_found(exc):
                    return None
                raise
            return Path(local).read_bytes()

        return _retry(go, what=f"download {path_in_repo}")

    def _list(self, path_in_repo: str, repo: str | None = None) -> list[str]:
        repo = repo or self.repo_id

        def go() -> list[str]:
            try:
                entries = self.api.list_repo_tree(
                    repo, path_in_repo=path_in_repo, repo_type="dataset", recursive=False
                )
                return [e.path for e in entries]
            except Exception as exc:
                if _is_not_found(exc):
                    return []
                raise

        return _retry(go, what=f"list {path_in_repo}")

    @staticmethod
    def _seqs(paths: list[str]) -> list[int]:
        out = []
        for p in paths:
            m = _SEQ_RE.search(p)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    # -- local (producer) side -------------------------------------------
    def start_run(self, run_id: str, info: dict[str, Any]) -> None:
        doc = dict(info, run_id=run_id, status="active", started_at=time.time())
        self._upload(
            f"runs/{run_id}/run.json",
            json.dumps(doc, indent=1).encode("utf-8"),
            f"start run {run_id}",
        )

    def finish_run(self, run_id: str, summary: dict[str, Any] | None = None) -> None:
        self._upload(
            f"runs/{run_id}/DONE", json.dumps(summary or {}).encode("utf-8"), f"finish run {run_id}"
        )

    def put_job(self, run_id: str, seq: int, payload: bytes) -> None:
        self._upload(f"runs/{run_id}/jobs/{seq:05d}.npz", payload, f"{run_id} job {seq}",
                     repo=self.job_repos[seq % N_SHARDS])

    def get_plan(self, run_id: str, seq: int) -> str | None:
        data = self._download(f"runs/{run_id}/plans/{seq:05d}.json", mutable=False,
                              repo=self.plan_repos[seq % N_SHARDS])
        return None if data is None else data.decode("utf-8")

    def heartbeat(self) -> dict[str, Any] | None:
        data = self._download("worker/heartbeat.json", mutable=True)
        return None if data is None else json.loads(data.decode("utf-8"))

    def request_stop(self) -> None:
        self._upload("worker/STOP", b"stop", "request worker stop")

    def publish_code(self, acarla_src: str | Path) -> bool:
        """Push `types.py` + `loop/*` as one JSON document so a freshly started
        worker runs the same codec/transport as this side. Skipped (no commit)
        when the published copy is already identical."""
        root = Path(acarla_src)
        files = {"acarla/__init__.py": ""}
        for rel in ("types.py", "loop/__init__.py", "loop/codec.py", "loop/transport.py"):
            files[f"acarla/{rel}"] = (root / rel).read_text(encoding="utf-8")
        payload = json.dumps(files, sort_keys=True).encode("utf-8")
        current = self._download("code/acarla_pkg.json", mutable=True)
        if current == payload:
            return False
        self._upload("code/acarla_pkg.json", payload, "publish acarla loop package")
        return True

    # -- worker (consumer) side ------------------------------------------
    def list_runs(self) -> list[str]:
        return sorted(p.rsplit("/", 1)[-1] for p in self._list("runs"))

    def run_is_done(self, run_id: str) -> bool:
        return any(p.endswith("/DONE") for p in self._list(f"runs/{run_id}"))

    def pending_jobs(self, run_id: str) -> list[int]:
        jobs: set[int] = set()
        for repo in self.job_repos:
            jobs |= set(self._seqs(self._list(f"runs/{run_id}/jobs", repo=repo)))
        plans: set[int] = set()
        for repo in self.plan_repos:
            plans |= set(self._seqs(self._list(f"runs/{run_id}/plans", repo=repo)))
        return sorted(jobs - plans)

    def get_job(self, run_id: str, seq: int) -> bytes:
        data = self._download(f"runs/{run_id}/jobs/{seq:05d}.npz", mutable=False,
                              repo=self.job_repos[seq % N_SHARDS])
        if data is None:
            raise FileNotFoundError(f"runs/{run_id}/jobs/{seq:05d}.npz")
        return data

    def put_plan(self, run_id: str, seq: int, text: str) -> None:
        self._upload(f"runs/{run_id}/plans/{seq:05d}.json", text.encode("utf-8"),
                     f"{run_id} plan {seq}", repo=self.plan_repos[seq % N_SHARDS])

    def write_heartbeat(self, state: dict[str, Any]) -> None:
        doc = dict(state, ts=time.time())
        self._upload("worker/heartbeat.json", json.dumps(doc).encode("utf-8"), "heartbeat")

    def stop_requested(self) -> bool:
        return any(p.endswith("/STOP") for p in self._list("worker"))

    def clear_stop(self) -> None:
        try:
            _retry(
                lambda: self.api.delete_file(
                    "worker/STOP", self.repo_id, repo_type="dataset", commit_message="clear STOP"
                ),
                what="delete STOP",
            )
        except Exception:
            pass


class LocalDirQueue:
    """Same interface as `LoopQueue`, backed by a directory. For tests and
    for smoke-testing the CARLA side against `scripts/fake_loop_worker.py`
    without Kaggle or a Hugging Face login. Selected via repo id `dir:<path>`."""

    def __init__(self, root: str | Path) -> None:
        self.repo_id = f"dir:{root}"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_repo(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def whoami(self) -> str:
        return "local"

    def _write(self, rel: str, data: bytes) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic on the same filesystem: readers never see partial files

    def _read(self, rel: str) -> bytes | None:
        path = self.root / rel
        return path.read_bytes() if path.exists() else None

    def _list(self, rel: str) -> list[str]:
        d = self.root / rel
        if not d.is_dir():
            return []
        return [f"{rel}/{p.name}" for p in d.iterdir() if not p.name.endswith(".part")]

    _seqs = staticmethod(LoopQueue._seqs)

    def start_run(self, run_id: str, info: dict[str, Any]) -> None:
        doc = dict(info, run_id=run_id, status="active", started_at=time.time())
        self._write(f"runs/{run_id}/run.json", json.dumps(doc, indent=1).encode("utf-8"))

    def finish_run(self, run_id: str, summary: dict[str, Any] | None = None) -> None:
        self._write(f"runs/{run_id}/DONE", json.dumps(summary or {}).encode("utf-8"))

    def put_job(self, run_id: str, seq: int, payload: bytes) -> None:
        self._write(f"runs/{run_id}/jobs/{seq:05d}.npz", payload)

    def get_plan(self, run_id: str, seq: int) -> str | None:
        data = self._read(f"runs/{run_id}/plans/{seq:05d}.json")
        return None if data is None else data.decode("utf-8")

    def heartbeat(self) -> dict[str, Any] | None:
        data = self._read("worker/heartbeat.json")
        return None if data is None else json.loads(data.decode("utf-8"))

    def request_stop(self) -> None:
        self._write("worker/STOP", b"stop")

    def list_runs(self) -> list[str]:
        return sorted(p.rsplit("/", 1)[-1] for p in self._list("runs"))

    def run_is_done(self, run_id: str) -> bool:
        return (self.root / f"runs/{run_id}/DONE").exists()

    def pending_jobs(self, run_id: str) -> list[int]:
        jobs = set(self._seqs(self._list(f"runs/{run_id}/jobs")))
        plans = set(self._seqs(self._list(f"runs/{run_id}/plans")))
        return sorted(jobs - plans)

    def get_job(self, run_id: str, seq: int) -> bytes:
        data = self._read(f"runs/{run_id}/jobs/{seq:05d}.npz")
        if data is None:
            raise FileNotFoundError(f"runs/{run_id}/jobs/{seq:05d}.npz")
        return data

    def put_plan(self, run_id: str, seq: int, text: str) -> None:
        self._write(f"runs/{run_id}/plans/{seq:05d}.json", text.encode("utf-8"))

    def write_heartbeat(self, state: dict[str, Any]) -> None:
        self._write("worker/heartbeat.json", json.dumps(dict(state, ts=time.time())).encode("utf-8"))

    def stop_requested(self) -> bool:
        return (self.root / "worker/STOP").exists()

    def clear_stop(self) -> None:
        try:
            (self.root / "worker/STOP").unlink()
        except FileNotFoundError:
            pass


def open_queue(repo_id: str, cache_dir: str | Path | None = None) -> LoopQueue | LocalDirQueue:
    """`dir:<path>` -> LocalDirQueue, anything else -> Hugging Face repo."""
    if repo_id.startswith("dir:"):
        return LocalDirQueue(repo_id[4:])
    return LoopQueue(repo_id, cache_dir=cache_dir)
