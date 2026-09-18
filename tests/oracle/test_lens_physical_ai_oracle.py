"""Oracle test for `src/acarla/adapter/lens.py::FThetaModel` against the
real `physical_ai_av.utils.camera_models.FThetaCameraModel`.

Runs ONLY in `.venv-sim`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

physical_ai_av = pytest.importorskip("physical_ai_av")
pd = pytest.importorskip("pandas")

from physical_ai_av.utils.camera_models import FThetaCameraModel  # noqa: E402

from acarla.adapter.lens import FThetaModel  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "m2_reference"


def _load_all_intrinsics_rows() -> list[dict]:
    with open(GOLDEN_DIR / "calibration_camera_intrinsics.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", _load_all_intrinsics_rows(), ids=lambda r: r["camera_name"])
def test_ftheta_port_matches_physical_ai_av(row):
    series = pd.Series({k: float(v) if k != "camera_name" else v for k, v in row.items()})
    reference = FThetaCameraModel.from_camera_row(series)

    ours = FThetaModel(
        width=int(row["width"]),
        height=int(row["height"]),
        cx=float(row["cx"]),
        cy=float(row["cy"]),
        fw_coef=np.array([float(row[f"fw_poly_{i}"]) for i in range(5)]),
        bw_coef=np.array([float(row[f"bw_poly_{i}"]) for i in range(5)]),
    )

    rng = np.random.default_rng(hash(row["camera_name"]) % (2**32))
    n = 1000
    theta = rng.uniform(0, np.radians(60), size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    rays = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )

    pixels_ref = reference.ray2pixel(rays)
    pixels_ours = ours.ray2pixel(rays)
    assert np.allclose(pixels_ref, pixels_ours, atol=1e-9)

    rays_back_ref = reference.pixel2ray(pixels_ref)
    rays_back_ours = ours.pixel2ray(pixels_ours)
    assert np.allclose(rays_back_ref, rays_back_ours, atol=1e-9)
