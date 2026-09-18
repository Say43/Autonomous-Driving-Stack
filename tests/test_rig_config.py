"""Tests for acarla.sim.rig and configs/rig_provisional.yaml.

Runs in the carla-less `.venv` (Python 3.13) -- acarla.sim.rig may not
import carla.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parent.parent
RIG_YAML = ROOT / "configs" / "rig_provisional.yaml"

from acarla.sim.rig import CameraSpec, load_rig_config  # noqa: E402
from acarla.types import ALPAMAYO_CAMERA_INDICES  # noqa: E402


def test_no_carla_import() -> None:
    assert "carla" not in sys.modules


def test_rig_provisional_file_exists_and_has_disclaimer() -> None:
    assert RIG_YAML.exists()
    text = RIG_YAML.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "provisor" in lowered
    assert "m1" in lowered


def test_rig_provisional_loads_all_four_default_cameras() -> None:
    specs = load_rig_config(RIG_YAML)
    names = [s.name for s in specs]
    assert set(names) == {"front_wide", "front_tele", "cross_left", "cross_right"}
    for spec in specs:
        assert spec.name in ALPAMAYO_CAMERA_INDICES
        assert isinstance(spec, CameraSpec)
        assert spec.width > 0 and spec.height > 0
        assert spec.fov > 0
        assert spec.sensor_tick > 0


def test_rig_provisional_camera_order_matches_alpamayo_index_order() -> None:
    specs = load_rig_config(RIG_YAML)
    names = [s.name for s in specs]
    # cross_left(0), front_wide(1), cross_right(2), front_tele(6)
    assert names == ["cross_left", "front_wide", "cross_right", "front_tele"]


def test_two_camera_subset_via_cli_option() -> None:
    specs = load_rig_config(RIG_YAML, cameras=["front_wide", "cross_left"])
    names = [s.name for s in specs]
    assert names == ["cross_left", "front_wide"]


def test_requesting_unknown_camera_raises() -> None:
    with pytest.raises(ValueError, match="not known Alpamayo camera"):
        load_rig_config(RIG_YAML, cameras=["front_wide", "rear_left"])


def test_requesting_camera_not_defined_in_file_raises(tmp_path: Path) -> None:
    minimal = tmp_path / "minimal_rig.yaml"
    minimal.write_text(
        dedent(
            """\
            cameras:
              front_wide:
                fov: 120
                width: 100
                height: 100
                sensor_tick: 0.1
                transform:
                  location: {x: 1.0, y: 0.0, z: 1.0}
                  rotation: {roll: 0.0, pitch: 0.0, yaw: 0.0}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not defined in this rig file"):
        load_rig_config(minimal, cameras=["front_wide", "cross_left"])


def test_rejects_unknown_camera_name_in_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad_rig.yaml"
    bad.write_text(
        dedent(
            """\
            cameras:
              rear_left:
                fov: 70
                width: 100
                height: 100
                sensor_tick: 0.1
                transform:
                  location: {x: -1.0, y: 0.0, z: 1.0}
                  rotation: {roll: 0.0, pitch: 0.0, yaw: 180.0}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not known Alpamayo camera"):
        load_rig_config(bad)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "missing_key.yaml"
    bad.write_text(
        dedent(
            """\
            cameras:
              front_wide:
                fov: 120
                width: 100
                height: 100
                transform:
                  location: {x: 1.0, y: 0.0, z: 1.0}
                  rotation: {roll: 0.0, pitch: 0.0, yaw: 0.0}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sensor_tick"):
        load_rig_config(bad)


def test_missing_transform_axis_raises(tmp_path: Path) -> None:
    bad = tmp_path / "missing_axis.yaml"
    bad.write_text(
        dedent(
            """\
            cameras:
              front_wide:
                fov: 120
                width: 100
                height: 100
                sensor_tick: 0.1
                transform:
                  location: {x: 1.0, y: 0.0}
                  rotation: {roll: 0.0, pitch: 0.0, yaw: 0.0}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="z"):
        load_rig_config(bad)


def test_empty_cameras_mapping_raises(tmp_path: Path) -> None:
    bad = tmp_path / "empty.yaml"
    bad.write_text("cameras: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_rig_config(bad)


def test_missing_cameras_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_cameras_key.yaml"
    bad.write_text("not_cameras: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cameras"):
        load_rig_config(bad)
