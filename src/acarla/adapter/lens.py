"""Lens/camera models used to turn CARLA's pinhole renders into images that
match the F-Theta fisheye cameras of the Alpamayo/PhysicalAI-AV dataset.

Pure numpy/scipy -- no `physical_ai_av` or `pandas` dependency, so this can
run in the 3.13 test environment. `FThetaModel` is a straight numeric port
of `physical_ai_av.utils.camera_models.FThetaCameraModel` (the same ~20
lines of math); the port is checked against the real class in
`tests/oracle/test_lens_physical_ai_oracle.py`.

All models use the optical (OpenCV) camera convention: x right in image, y
down in image, z forward (viewing direction) -- the same convention
`src/acarla/sim/coords.py` converts to/from the rig frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class FThetaModel:
    """F-Theta fisheye camera model: pixel radius r = fw_poly(theta), theta
    = angle to the optical axis. Numeric port of
    `physical_ai_av.utils.camera_models.FThetaCameraModel`.
    """

    width: int
    height: int
    cx: float
    cy: float
    fw_coef: np.ndarray  # (5,) low-to-high degree, theta -> r
    bw_coef: np.ndarray  # (5,) low-to-high degree, r -> theta

    def __post_init__(self) -> None:
        self.fw_coef = np.asarray(self.fw_coef, dtype=float)
        self.bw_coef = np.asarray(self.bw_coef, dtype=float)
        if self.fw_coef.shape != (5,) or self.bw_coef.shape != (5,):
            raise ValueError(
                "FThetaModel: fw_coef and bw_coef must both have shape (5,), "
                f"got {self.fw_coef.shape} and {self.bw_coef.shape}"
            )

    def _fw_poly(self, theta: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(theta, self.fw_coef)

    def _bw_poly(self, r: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(r, self.bw_coef)

    def ray2pixel(self, rays: np.ndarray) -> np.ndarray:
        """Rays [..., 3] (optical frame) -> pixel coordinates [..., 2]."""
        rays = np.asarray(rays, dtype=float)
        theta = np.arccos(rays[..., 2] / np.linalg.norm(rays, axis=-1))
        xy_norm = np.linalg.norm(rays[..., :2], axis=-1)
        r = self._fw_poly(theta)
        principal = np.array([self.cx, self.cy])
        return principal + r[..., None] * rays[..., :2] / xy_norm[..., None]

    def pixel2ray(self, pixels: np.ndarray) -> np.ndarray:
        """Pixel coordinates [..., 2] -> unit rays [..., 3] (optical frame)."""
        pixels = np.asarray(pixels, dtype=float)
        p = pixels - np.array([self.cx, self.cy])
        r = np.linalg.norm(p, axis=-1)
        theta = self._bw_poly(r)
        c, s = np.cos(theta), np.sin(theta)
        ray = np.stack([s * p[..., 0] / r, s * p[..., 1] / r, c], axis=-1)
        return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


@dataclass
class PinholeModel:
    """Rectilinear pinhole camera model, optical (OpenCV) convention."""

    width: int
    height: int
    hfov_deg: float

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def focal(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    def ray2pixel(self, rays: np.ndarray) -> np.ndarray:
        rays = np.asarray(rays, dtype=float)
        f = self.focal
        x = f * rays[..., 0] / rays[..., 2] + self.cx
        y = f * rays[..., 1] / rays[..., 2] + self.cy
        return np.stack([x, y], axis=-1)

    def pixel2ray(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float)
        f = self.focal
        x = (pixels[..., 0] - self.cx) / f
        y = (pixels[..., 1] - self.cy) / f
        z = np.ones_like(x)
        ray = np.stack([x, y, z], axis=-1)
        return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def _boundary_pixels(ftheta: FThetaModel, n: int = 400) -> np.ndarray:
    """Sample points along the four edges of the F-Theta image. The
    backward polynomial's pixel radius is monotonic in theta, so the
    largest angle to the optical axis (in any fixed direction) over the
    whole image is attained somewhere on the boundary; sampling the edges
    densely is sufficient without walking every interior pixel.
    """
    width, height = ftheta.width, ftheta.height
    xs = np.linspace(0, width - 1, n)
    ys = np.linspace(0, height - 1, n)
    top = np.stack([xs, np.zeros(n)], axis=-1)
    bottom = np.stack([xs, np.full(n, height - 1)], axis=-1)
    left = np.stack([np.zeros(n), ys], axis=-1)
    right = np.stack([np.full(n, width - 1), ys], axis=-1)
    return np.concatenate([top, bottom, left, right], axis=0)


def max_theta_deg(ftheta: FThetaModel) -> float:
    """Largest angle-to-optical-axis (degrees) still inside the F-Theta
    image, evaluated along the image boundary via the backward polynomial.
    """
    pixels = _boundary_pixels(ftheta)
    p = pixels - np.array([ftheta.cx, ftheta.cy])
    r = np.linalg.norm(p, axis=-1)
    theta = ftheta._bw_poly(r)
    return float(np.degrees(np.max(theta)))


def required_pinhole_hfov_deg(ftheta: FThetaModel, margin_deg: float = 2.0) -> float:
    """Horizontal pinhole FOV (degrees) such that every F-Theta boundary
    pixel (and, by monotonicity of the F-Theta radius, every interior
    pixel too), with `margin_deg` of slack, still projects inside a
    `width` x `height` pinhole image.

    Unlike `max_theta_deg` (an isotropic cone angle), a pinhole camera has
    independent horizontal and vertical half-angles, so this computes the
    per-axis angle (angle between the ray and the optical axis, measured
    within the horizontal/vertical plane respectively: `atan(|rx|/rz)` and
    `atan(|ry|/rz)`) for every boundary pixel, takes the worst case in each
    axis, adds the margin, and derives the shared focal length `f` (same
    for x and y, since both axes share one pinhole focal length) as the
    tighter of the two per-axis constraints. Using the isotropic cone angle
    for both axes would substantially overestimate the required horizontal
    FOV for a wide-aspect image, since the image corners are reached mostly
    horizontally, not vertically.

    Raises `ValueError` if the resulting horizontal FOV exceeds 150 degrees
    (a single pinhole camera can no longer represent the required field of
    view without extreme distortion/stretching).
    """
    pixels = _boundary_pixels(ftheta)
    rays = ftheta.pixel2ray(pixels)
    if np.any(rays[..., 2] <= 0):
        raise ValueError(
            "required_pinhole_hfov_deg: some boundary rays point behind the "
            "camera (rz <= 0) -- this F-Theta FOV exceeds 180 deg and cannot "
            "be represented by any pinhole camera"
        )
    angle_x = np.arctan(np.abs(rays[..., 0] / rays[..., 2]))
    angle_y = np.arctan(np.abs(rays[..., 1] / rays[..., 2]))
    half_hfov_needed = angle_x.max() + np.radians(margin_deg)
    half_vfov_needed = angle_y.max() + np.radians(margin_deg)

    if half_hfov_needed >= np.pi / 2 or half_vfov_needed >= np.pi / 2:
        raise ValueError(
            "required_pinhole_hfov_deg: required half-angle "
            f"(x={np.degrees(half_hfov_needed):.2f} deg, "
            f"y={np.degrees(half_vfov_needed):.2f} deg) >= 90 deg, no "
            "pinhole camera can cover this F-Theta field of view"
        )

    f_x = (ftheta.width / 2.0) / np.tan(half_hfov_needed)
    f_y = (ftheta.height / 2.0) / np.tan(half_vfov_needed)
    f = min(f_x, f_y)  # the tighter constraint sets the shared focal length
    hfov = 2.0 * np.degrees(np.arctan((ftheta.width / 2.0) / f))

    if hfov > 150.0:
        raise ValueError(
            f"required_pinhole_hfov_deg: required horizontal FOV {hfov:.2f} "
            "deg exceeds 150 deg -- a single pinhole camera cannot cover "
            f"the F-Theta field of view (max boundary angle x="
            f"{np.degrees(angle_x.max()):.2f} deg, y="
            f"{np.degrees(angle_y.max()):.2f} deg, margin={margin_deg} deg)"
        )
    return hfov


def build_remap(
    ftheta: FThetaModel, pinhole: PinholeModel
) -> tuple[np.ndarray, np.ndarray]:
    """For every F-Theta target pixel, compute the corresponding source
    pixel in the pinhole image. Returns (map_x, map_y), each (H, W) float32,
    H = ftheta.height, W = ftheta.width.
    """
    ys, xs = np.meshgrid(
        np.arange(ftheta.height, dtype=float),
        np.arange(ftheta.width, dtype=float),
        indexing="ij",
    )
    target_pixels = np.stack([xs, ys], axis=-1)
    rays = ftheta.pixel2ray(target_pixels)
    source_pixels = pinhole.ray2pixel(rays)
    map_x = source_pixels[..., 0].astype(np.float32)
    map_y = source_pixels[..., 1].astype(np.float32)
    return map_x, map_y


def apply_remap(image_hwc_uint8: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Resample `image_hwc_uint8` (H, W, C) at (map_x, map_y) per output
    pixel via bilinear interpolation (order=1). Output has the shape of
    `map_x`/`map_y` with the same channel count; out-of-bounds source
    coordinates map to 0.
    """
    image = np.asarray(image_hwc_uint8)
    out_h, out_w = map_x.shape
    n_channels = image.shape[-1]
    coords = np.stack([map_y, map_x], axis=0)  # (row, col) order for map_coordinates
    out = np.zeros((out_h, out_w, n_channels), dtype=np.float64)
    for c in range(n_channels):
        out[..., c] = map_coordinates(
            image[..., c].astype(np.float64),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
    return np.clip(out, 0, 255).astype(np.uint8)
