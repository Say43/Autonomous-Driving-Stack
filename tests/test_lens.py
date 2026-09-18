"""Non-oracle tests for `src/acarla/adapter/lens.py`."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from acarla.adapter.lens import (
    FThetaModel,
    PinholeModel,
    apply_remap,
    build_remap,
    max_theta_deg,
    required_pinhole_hfov_deg,
)

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden" / "m2_reference"

# Calibration data extracted from the PhysicalAI-AV dataset is licensed for internal
# use only (NVIDIA AV Dataset License, section 4.6) and is therefore not part of the
# public repository; regenerate it with the notebooks/scripts if you hold a licence.
pytestmark = pytest.mark.skipif(
    not (GOLDEN_DIR / "calibration_camera_intrinsics.csv").exists(),
    reason="private PhysicalAI-AV calibration CSVs not present",
)


def _load_intrinsics_row(camera_name: str) -> dict:
    with open(GOLDEN_DIR / "calibration_camera_intrinsics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["camera_name"] == camera_name:
                return row
    raise KeyError(camera_name)


def _ftheta_from_row(row: dict) -> FThetaModel:
    return FThetaModel(
        width=int(row["width"]),
        height=int(row["height"]),
        cx=float(row["cx"]),
        cy=float(row["cy"]),
        fw_coef=np.array([float(row[f"fw_poly_{i}"]) for i in range(5)]),
        bw_coef=np.array([float(row[f"bw_poly_{i}"]) for i in range(5)]),
    )


@pytest.fixture
def front_wide() -> FThetaModel:
    return _ftheta_from_row(_load_intrinsics_row("camera_front_wide_120fov"))


def _random_rays_within_fov(n, max_theta_deg_, rng):
    theta = rng.uniform(0, np.radians(max_theta_deg_ * 0.95), size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def test_ftheta_pixel_ray_roundtrip(front_wide):
    rng = np.random.default_rng(0)
    mtheta = max_theta_deg(front_wide)
    rays = _random_rays_within_fov(500, mtheta, rng)
    pixels = front_wide.ray2pixel(rays)
    rays_back = front_wide.pixel2ray(pixels)
    # fw_poly/bw_poly are independently fitted polynomials (not exact
    # analytic inverses), so the roundtrip through real calibration data
    # has small residual error -- verified against physical_ai_av's own
    # FThetaCameraModel in tests/oracle/test_lens_physical_ai_oracle.py to
    # show our port reproduces that same residual, not a bug of the port.
    assert np.allclose(rays, rays_back, atol=1e-4)


def test_pinhole_pixel_ray_roundtrip():
    pinhole = PinholeModel(width=1920, height=1080, hfov_deg=90.0)
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(0, pinhole.width, size=n)
    y = rng.uniform(0, pinhole.height, size=n)
    pixels = np.stack([x, y], axis=-1)
    rays = pinhole.pixel2ray(pixels)
    pixels_back = pinhole.ray2pixel(rays)
    assert np.allclose(pixels, pixels_back, atol=1e-6)


def test_required_pinhole_hfov_covers_max_theta(front_wide):
    hfov = required_pinhole_hfov_deg(front_wide, margin_deg=2.0)
    assert 0 < hfov < 150
    pinhole = PinholeModel(width=front_wide.width, height=front_wide.height, hfov_deg=hfov)
    # Every boundary pixel of the F-Theta image must project inside the
    # pinhole image (with slack for the margin already baked into hfov).
    boundary = np.array(
        [[0.0, 0.0], [front_wide.width - 1, 0.0], [0.0, front_wide.height - 1],
         [front_wide.width - 1, front_wide.height - 1],
         [front_wide.width / 2, 0.0], [front_wide.width / 2, front_wide.height - 1],
         [0.0, front_wide.height / 2], [front_wide.width - 1, front_wide.height / 2]]
    )
    rays = front_wide.pixel2ray(boundary)
    pixels_in_pinhole = pinhole.ray2pixel(rays)
    assert np.all(pixels_in_pinhole[:, 0] >= 0)
    assert np.all(pixels_in_pinhole[:, 0] <= pinhole.width)
    assert np.all(pixels_in_pinhole[:, 1] >= 0)
    assert np.all(pixels_in_pinhole[:, 1] <= pinhole.height)


def test_required_pinhole_hfov_raises_for_extreme_fov():
    # A synthetic F-Theta model whose backward polynomial reports a huge
    # theta at the corners -- larger than any pinhole can represent.
    ftheta = FThetaModel(
        width=1920,
        height=1080,
        cx=960,
        cy=540,
        fw_coef=np.array([0.0, 500.0, 0.0, 0.0, 0.0]),
        bw_coef=np.array([0.0, 1.0 / 300.0, 0.0, 0.0, 0.0]),
    )
    with pytest.raises(ValueError):
        required_pinhole_hfov_deg(ftheta, margin_deg=2.0)


def test_build_remap_maps_within_pinhole_bounds(front_wide):
    hfov = required_pinhole_hfov_deg(front_wide, margin_deg=2.0)
    pinhole = PinholeModel(width=front_wide.width, height=front_wide.height, hfov_deg=hfov)
    map_x, map_y = build_remap(front_wide, pinhole)
    assert map_x.shape == (front_wide.height, front_wide.width)
    assert map_y.shape == (front_wide.height, front_wide.width)
    assert np.all(np.isfinite(map_x))
    assert np.all(np.isfinite(map_y))
    assert map_x.min() >= -1.0
    assert map_x.max() <= pinhole.width + 1.0
    assert map_y.min() >= -1.0
    assert map_y.max() <= pinhole.height + 1.0


def test_apply_remap_bright_point_lands_at_expected_pixel(front_wide):
    hfov = required_pinhole_hfov_deg(front_wide, margin_deg=2.0)
    pinhole = PinholeModel(width=front_wide.width, height=front_wide.height, hfov_deg=hfov)

    # Pick a ray well within the FOV, find where it lands in the pinhole
    # source image, paint a bright point there, remap, and check it lands
    # where ftheta.ray2pixel says it should.
    ray = np.array([0.15, -0.08, 1.0])
    ray = ray / np.linalg.norm(ray)
    src_px = pinhole.ray2pixel(ray)
    src_x, src_y = int(round(src_px[0])), int(round(src_px[1]))

    image = np.zeros((pinhole.height, pinhole.width, 3), dtype=np.uint8)
    image[src_y, src_x, :] = 255

    map_x, map_y = build_remap(front_wide, pinhole)
    out = apply_remap(image, map_x, map_y)

    target_px = front_wide.ray2pixel(ray)
    ty, tx = int(round(target_px[1])), int(round(target_px[0]))

    # Find brightest pixel near the expected location.
    window = out[max(0, ty - 3) : ty + 4, max(0, tx - 3) : tx + 4, :].sum(axis=-1)
    assert window.max() > 100
    py, px = np.unravel_index(np.argmax(window), window.shape)
    found_y = max(0, ty - 3) + py
    found_x = max(0, tx - 3) + px
    assert abs(found_x - tx) <= 1
    assert abs(found_y - ty) <= 1
