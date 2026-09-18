"""Tests for the M1-calibrated rig format (`configs/rig_alpamayo.yaml`).

Runs in the carla-less `.venv` (Python 3.13) -- `acarla.sim.rig` may not
import carla.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RIG_YAML = ROOT / "configs" / "rig_alpamayo.yaml"

# Calibration data extracted from the PhysicalAI-AV dataset is licensed for internal
# use only (NVIDIA AV Dataset License, section 4.6) and is therefore not part of the
# public repository; regenerate it with the notebooks/scripts if you hold a licence.
pytestmark = pytest.mark.skipif(
    not RIG_YAML.exists(), reason="private calibrated rig (configs/rig_alpamayo.yaml) not present"
)

from acarla.adapter.lens import FThetaModel  # noqa: E402
from acarla.sim.rig import (  # noqa: E402
    CameraSpec,
    load_rig_config,
    load_rig_intrinsics,
    load_rig_vehicle,
)


def test_no_carla_import() -> None:
    assert "carla" not in sys.modules


def test_calibrated_rig_loads_all_four_cameras() -> None:
    specs = load_rig_config(RIG_YAML)
    names = [s.name for s in specs]
    assert set(names) == {"cross_left", "front_wide", "cross_right", "front_tele"}
    for spec in specs:
        assert isinstance(spec, CameraSpec)
        assert spec.width == 1920
        assert spec.height == 1080
        assert spec.sensor_tick == pytest.approx(0.1)


def test_calibrated_rig_camera_order_matches_alpamayo_index_order() -> None:
    specs = load_rig_config(RIG_YAML)
    names = [s.name for s in specs]
    assert names == ["cross_left", "front_wide", "cross_right", "front_tele"]


def test_front_wide_matches_yaml_values() -> None:
    """The loader must hand the YAML values through untouched. The expected
    values are read from the same file instead of being spelled out here: the
    calibration numbers are dataset-derived and stay out of the source tree."""
    doc = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
    cam = doc["cameras"]["front_wide"]["carla"]
    specs = load_rig_config(RIG_YAML)
    front_wide = next(s for s in specs if s.name == "front_wide")
    assert front_wide.fov == pytest.approx(cam["render_hfov_deg"])
    assert front_wide.x == pytest.approx(cam["location"]["x"])
    assert front_wide.y == pytest.approx(cam["location"]["y"])
    assert front_wide.z == pytest.approx(cam["location"]["z"])
    assert front_wide.pitch == pytest.approx(cam["rotation"]["pitch"])
    assert front_wide.yaw == pytest.approx(cam["rotation"]["yaw"])
    assert 100.0 < front_wide.fov < 150.0


def test_load_rig_intrinsics_returns_ftheta_models_for_all_cameras() -> None:
    intrinsics = load_rig_intrinsics(RIG_YAML)
    assert set(intrinsics) == {"cross_left", "front_wide", "cross_right", "front_tele"}
    for model in intrinsics.values():
        assert isinstance(model, FThetaModel)
        assert model.width == 1920
        assert model.height == 1080


def test_load_rig_vehicle_returns_expected_keys() -> None:
    vehicle = load_rig_vehicle(RIG_YAML)
    assert 0.5 < vehicle["rear_axle_to_bbox_center"] < 3.0  # value itself is dataset-derived
    assert "wheelbase" in vehicle
    assert "height" in vehicle


def test_two_camera_subset_via_cli_option() -> None:
    specs = load_rig_config(RIG_YAML, cameras=["front_wide", "cross_left"])
    names = [s.name for s in specs]
    assert names == ["cross_left", "front_wide"]
