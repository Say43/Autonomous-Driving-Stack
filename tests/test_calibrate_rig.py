"""Tests for scripts/calibrate_rig.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Calibration data extracted from the PhysicalAI-AV dataset is licensed for internal
# use only (NVIDIA AV Dataset License, section 4.6) and is therefore not part of the
# public repository; regenerate it with the notebooks/scripts if you hold a licence.
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "golden" / "m2_reference" / "calibration_sensor_extrinsics.csv").exists(),
    reason="private PhysicalAI-AV calibration CSVs not present",
)


def test_calibrate_rig_script_produces_valid_config(tmp_path):
    out_path = tmp_path / "rig_alpamayo.yaml"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_rig.py"),
            "--calibration-dir",
            str(REPO_ROOT / "golden" / "m2_reference"),
            "--out",
            str(out_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    with open(out_path) as f:
        config = yaml.safe_load(f)

    expected_cams = {"cross_left", "front_wide", "cross_right", "front_tele"}
    assert set(config["cameras"].keys()) == expected_cams

    front_wide_yaw = config["cameras"]["front_wide"]["carla"]["rotation"]["yaw"]
    assert -2.0 <= front_wide_yaw <= 2.0

    yaw_l = config["cameras"]["cross_left"]["carla"]["rotation"]["yaw"]
    yaw_r = config["cameras"]["cross_right"]["carla"]["rotation"]["yaw"]
    assert (yaw_l < 0) != (yaw_r < 0)
    assert 60 <= abs(yaw_l) <= 75
    assert 60 <= abs(yaw_r) <= 75

    for cam in config["cameras"].values():
        assert cam["carla"]["render_hfov_deg"] < 150

    assert "vehicle" in config
    assert "rig_origin_note" in config
    assert config["carla_attach_offset_open_issue"] is True
