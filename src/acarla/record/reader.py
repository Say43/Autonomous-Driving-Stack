"""Streaming JSONL trace reader.

No dependency on `carla`. Mirrors `acarla.record.writer.TraceWriter`: reads
the `RunHeader` first line, then yields `TraceFrame` records one at a time
without loading the whole file into memory.

A malformed line (bad JSON, or JSON that does not match the `TraceFrame`
contract in `acarla.types`) is skipped and counted rather than aborting the
read -- a single corrupted tick (e.g. from a crash mid-write) should not
throw away an otherwise-usable run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from acarla.types import RunHeader, TraceFrame


class TraceFrameIterator:
    """Iterates `TraceFrame` records from an open trace file.

    Malformed lines are skipped and counted in `skipped_lines` /
    `skipped_details` rather than raising -- callers that care can inspect
    those after exhausting the iterator. The underlying file is closed
    automatically once iteration completes (or via `close()` early)."""

    def __init__(self, file: IO[str], path: Path) -> None:
        self._file = file
        self._path = path
        self._closed = False
        self.skipped_lines = 0
        self.skipped_details: list[tuple[int, str]] = []

    def __iter__(self) -> TraceFrameIterator:
        return self

    def __next__(self) -> TraceFrame:
        while True:
            line = self._file.readline()
            if line == "":
                self.close()
                raise StopIteration
            line = line.strip()
            if not line:
                continue
            try:
                frame = TraceFrame.from_json_dict(json.loads(line))
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
                # malformed line (bad JSON, missing/mistyped fields, a shape
                # or rotation-matrix violation raised by TraceFrame's own
                # __post_init__ chain) is a skip, not an abort.
                self.skipped_lines += 1
                self.skipped_details.append((self.skipped_lines, str(exc)))
                continue
            return frame

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True


def read_trace(path: str | Path) -> tuple[RunHeader, TraceFrameIterator]:
    """Read a trace file's `RunHeader` and return it along with an iterator
    over its `TraceFrame` records.

    Raises `ValueError` if the file is empty or its first line does not
    parse as a `RunHeader`. Per-frame corruption after that point is
    reported through the returned iterator's `skipped_lines`, not raised.
    """
    path = Path(path)
    f = path.open("r", encoding="utf-8")
    header_line = f.readline()
    if not header_line.strip():
        f.close()
        raise ValueError(f"{path}: empty trace file, expected a RunHeader on line 1")

    try:
        header = RunHeader.from_json_dict(json.loads(header_line))
    except Exception as exc:
        f.close()
        raise ValueError(f"{path}: could not parse RunHeader on line 1: {exc}") from exc

    return header, TraceFrameIterator(f, path)
