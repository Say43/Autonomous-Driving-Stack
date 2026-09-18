"""Streaming JSONL trace writer.

No dependency on `carla`: this module only manages the trace file and the
on-disk image directory layout (as relative paths for `TraceFrame`), it
never encodes pixels itself. Image bytes are written by the caller (see
`scripts/run_open_loop.py`, which has `carla` available and uses
`carla.Image.save_to_disk` directly against the path this writer hands
back), so this module stays importable from the carla-less `.venv` and is
exercised directly by `tests/test_record_io.py`.

Writes stream to disk one line at a time -- nothing is buffered in memory
beyond the current frame, so long runs do not grow the writer's own memory
footprint.

Deviation from the project spec ("Bilder als JPEG"): images are written as
PNG, not JPEG. Verified against the actual M4 smoke-test run: this CARLA
0.9.16 Windows build's `carla.Image.save_to_disk(path)` writes PNG bytes
(magic `89 50 4E 47`, confirmed with `file`) regardless of whether `path`
ends in `.jpg` or `.png` -- it silently ignores the requested JPEG
extension rather than erroring, so shipping a `.jpg` name here would produce
files that are actually PNGs wearing the wrong extension. `.png` is what
this build can actually produce; naming the files accordingly is the
honest choice; JPEG re-encoding is not implemented in this milestone.
"""

from __future__ import annotations

import json
from pathlib import Path

from acarla.types import RunHeader, TraceFrame

IMAGES_DIRNAME = "images"


class TraceWriter:
    """Writes one JSONL trace file per run: a `RunHeader` first line, then
    one `TraceFrame` line per tick. Use as a context manager so the file is
    always closed, including on exception."""

    def __init__(self, out_dir: str | Path, header: RunHeader, image_ext: str = "png") -> None:
        if image_ext not in ("png", "jpg"):
            raise ValueError(f"image_ext must be png or jpg, got {image_ext!r}")
        self.image_ext = image_ext
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.out_dir / IMAGES_DIRNAME
        self.trace_path = self.out_dir / "trace.jsonl"

        self._file = self.trace_path.open("w", encoding="utf-8", newline="\n")
        self._closed = False
        self._n_frames_written = 0
        self._file.write(json.dumps(header.to_json_dict(), sort_keys=True))
        self._file.write("\n")
        self._file.flush()

    @property
    def n_frames_written(self) -> int:
        return self._n_frames_written

    def image_path(self, camera: str, frame_id: int) -> Path:
        """Absolute path where an image for `camera` at `frame_id` should be
        written. Creates the per-camera subdirectory on first use.

        `.png`, not `.jpg` -- see the module docstring: this CARLA build's
        `save_to_disk` writes PNG bytes regardless of the requested
        extension, so the extension here names what is actually produced.
        """
        cam_dir = self.images_dir / camera
        cam_dir.mkdir(parents=True, exist_ok=True)
        return cam_dir / f"{frame_id:06d}.{self.image_ext}"

    def relative_image_path(self, camera: str, frame_id: int) -> str:
        """The path stored in `TraceFrame.image_paths`, relative to
        `out_dir` -- always forward-slash separated regardless of OS, so
        trace files are portable."""
        return f"{IMAGES_DIRNAME}/{camera}/{frame_id:06d}.{self.image_ext}"

    def write_frame(self, frame: TraceFrame) -> None:
        if self._closed:
            raise RuntimeError(f"{self.trace_path}: writer is already closed")
        self._file.write(json.dumps(frame.to_json_dict(), sort_keys=True))
        self._file.write("\n")
        self._file.flush()
        self._n_frames_written += 1

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
