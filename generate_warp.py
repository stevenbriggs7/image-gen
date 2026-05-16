"""
Warp field post-processing: displace pixels of a rendered image.

The warp field is invisible — it remaps where pixels are sampled from,
producing organic distortion (push/pull) over any generator output.

Four warp types:
  Noise  — two independent fractal noise fields drive dx and dy;
            organic cloud-like distortion.
  Swirl  — pixels rotate by an angle that decays with distance from
            the canvas centre; vortex / whirlpool effect.
  Ripple — sine-wave displacement perpendicular to a chosen axis;
            heat-haze / fabric ripple.
  Radial — barrel (outward) or pincushion (inward) lens distortion.

Strength is specified in full-resolution pixels and scaled down
automatically for previews, keeping the visual effect consistent.
"""

import math
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from generate import _fractal_noise_field

DEFAULTS: dict = {
    "warp_type":   "none",   # "none" | "noise" | "swirl" | "ripple" | "radial"
    "strength":     0.0,     # intensity — pixels of max displacement (noise/ripple/swirl)
                             # or signed scale factor * 100 (radial)
    "noise_scale":  0.003,   # spatial frequency of the noise field
    "seed":        42,
    "falloff":      0.5,     # swirl: how fast rotation decays with radius (0=sharp edge, 1=gradual)
    "frequency":    3.0,     # ripple: cycles across the shorter canvas dimension
    "angle":        0.0,     # ripple: direction of the wave propagation (degrees)
}


# ── Displacement field builders ───────────────────────────────────────────────

def _noise_field(width: int, height: int, cfg: dict, scale: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    ns = float(cfg["noise_scale"]) / max(scale, 0.05)
    seed = int(cfg["seed"])
    strength = float(cfg["strength"]) * scale
    dx = _fractal_noise_field(width, height, ns, seed=seed).astype(np.float32) * strength
    dy = _fractal_noise_field(width, height, ns, seed=seed + 31337).astype(np.float32) * strength
    return dx, dy


def _swirl_field(width: int, height: int, cfg: dict, scale: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    strength = float(cfg["strength"]) * scale   # max rotation angle in degrees
    falloff  = float(cfg["falloff"])            # 0 = hard edge, 1 = very gradual
    cx, cy   = width / 2.0, height / 2.0
    max_r    = math.sqrt(cx ** 2 + cy ** 2)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    rx, ry = xx - cx, yy - cy
    r = np.sqrt(rx ** 2 + ry ** 2)

    # Angle decays from strength at centre to 0 at edge
    decay = np.exp(-falloff * 5.0 * r / (max_r + 1e-9))
    angle_rad = np.deg2rad(strength) * decay

    cos_a = np.cos(angle_rad) - 1.0   # delta from identity
    sin_a = np.sin(angle_rad)

    dx = rx * cos_a - ry * sin_a
    dy = rx * sin_a + ry * cos_a
    return dx.astype(np.float32), dy.astype(np.float32)


def _ripple_field(width: int, height: int, cfg: dict, scale: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    strength  = float(cfg["strength"]) * scale
    frequency = float(cfg["frequency"])
    angle_deg = float(cfg["angle"])
    angle_rad = math.radians(angle_deg)

    short = min(width, height)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # Project pixel coords onto the wave direction
    proj = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
    phase = (2.0 * math.pi * frequency / short) * proj

    # Displacement is perpendicular to the wave direction
    disp = np.sin(phase).astype(np.float32) * strength
    dx = disp * (-math.sin(angle_rad))
    dy = disp *   math.cos(angle_rad)
    return dx, dy


def _radial_field(width: int, height: int, cfg: dict, scale: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    # strength > 0 → barrel (push outward), < 0 → pincushion (pull inward)
    # Treat strength as a percentage scale factor on r^2 distortion.
    k = float(cfg["strength"]) / 100.0
    cx, cy  = width / 2.0, height / 2.0
    max_r2  = cx ** 2 + cy ** 2

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    rx, ry = xx - cx, yy - cy
    r2_norm = (rx ** 2 + ry ** 2) / (max_r2 + 1e-9)

    # Source coords: src = dst + delta; delta = k * r_norm^2 * (dst - centre)
    factor = (k * r2_norm).astype(np.float32)
    dx = rx * factor
    dy = ry * factor
    return dx, dy


# ── Public API ────────────────────────────────────────────────────────────────

def apply(img: Image.Image, cfg: dict, scale: float = 1.0) -> Image.Image:
    """
    Displace pixels of img according to the warp field described in cfg.

    Returns img unchanged when warp_type is 'none' or strength is 0.
    scale should match the scale used to render img (so strength stays
    visually consistent between preview and full-resolution export).
    """
    cfg = {**DEFAULTS, **cfg}
    kind = str(cfg["warp_type"]).lower()
    if kind == "none" or float(cfg["strength"]) == 0.0:
        return img

    width, height = img.size

    if kind == "noise":
        dx, dy = _noise_field(width, height, cfg, scale)
    elif kind == "swirl":
        dx, dy = _swirl_field(width, height, cfg, scale)
    elif kind == "ripple":
        dx, dy = _ripple_field(width, height, cfg, scale)
    elif kind == "radial":
        dx, dy = _radial_field(width, height, cfg, scale)
    else:
        return img

    arr = np.array(img, dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    src_y = (yy + dy).clip(0, height - 1)
    src_x = (xx + dx).clip(0, width  - 1)

    warped = np.empty_like(arr)
    for ch in range(3):
        warped[..., ch] = map_coordinates(
            arr[..., ch], [src_y, src_x], order=1, mode='nearest'
        )

    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGB")
