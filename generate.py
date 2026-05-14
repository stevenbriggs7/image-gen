"""
Core image generation: scattered lines guided by a Perlin-style flow field.

Mathematical principles:
  - Fractal value noise builds a smooth, multi-scale angle field (coherent
    randomness — locally structured, globally varied).
  - Line lengths follow a Gaussian distribution (natural spread around a mean).
  - Starting positions are uniform-random; the flow field gives them direction.
  - Alpha is sampled uniformly and composited onto the background analytically.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_lines": 1500,
    "noise_scale": 0.004,    # spatial frequency of the angle field
    "angle_range": 1.0,      # fraction of 2π covered by the field
    "length_median": 60.0,   # median line length in px (log-normal)
    "length_spread": 0.6,    # log-σ: 0.1 = uniform, 1.5 = extreme variation
    "margin": 0.0,           # fraction of each edge to exclude (0–0.45)
    "stroke_width_min": 1.0,
    "stroke_width_max": 2.5,
    "alpha_min": 50,         # 0–255
    "alpha_max": 180,
    "bg_dark": False,
}


# ── Noise field ────────────────────────────────────────────────────────────────

def _smooth_layer(width: int, height: int, grid_scale: float, seed: int) -> np.ndarray:
    """One octave of value noise: random coarse grid + smooth bilinear upsample."""
    gw = max(2, int(math.ceil(width * grid_scale)) + 2)
    gh = max(2, int(math.ceil(height * grid_scale)) + 2)
    rng = np.random.default_rng(seed)
    grid = rng.uniform(-1.0, 1.0, (gh, gw))

    # Fractional positions in grid space for every pixel
    gy = np.linspace(0.0, gh - 1.0, height)
    gx = np.linspace(0.0, gw - 1.0, width)

    iy = np.clip(np.floor(gy).astype(np.intp), 0, gh - 2)
    ix = np.clip(np.floor(gx).astype(np.intp), 0, gw - 2)
    fy = gy - iy
    fx = gx - ix

    # Smoothstep (Ken Perlin's fade)
    uy = fy * fy * (3.0 - 2.0 * fy)
    ux = fx * fx * (3.0 - 2.0 * fx)

    # Bilinear interpolation over the full 2-D field via outer-product indexing
    v00 = grid[np.ix_(iy,     ix    )]
    v10 = grid[np.ix_(iy,     ix + 1)]
    v01 = grid[np.ix_(iy + 1, ix    )]
    v11 = grid[np.ix_(iy + 1, ix + 1)]

    ux2d = ux[np.newaxis, :]   # (1, W)
    uy2d = uy[:, np.newaxis]   # (H, 1)

    return (v00 * (1 - ux2d) * (1 - uy2d)
            + v10 * ux2d      * (1 - uy2d)
            + v01 * (1 - ux2d) * uy2d
            + v11 * ux2d       * uy2d)


def _fractal_noise_field(
    width: int, height: int, noise_scale: float, seed: int,
    octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0,
) -> np.ndarray:
    """Sum of smooth layers at increasing frequencies → fractal noise (H, W)."""
    field = np.zeros((height, width))
    amp, total, freq = 1.0, 0.0, noise_scale
    for k in range(octaves):
        field += _smooth_layer(width, height, freq, seed + k * 7919) * amp
        total += amp
        amp *= persistence
        freq *= lacunarity
    return field / total


# ── Main generation entry point ────────────────────────────────────────────────

def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render scattered lines and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews)
    while preserving the spatial character of the noise field.
    """
    cfg = {**DEFAULTS, **config}

    seed     = int(cfg["seed"])
    width    = max(10, int(cfg["output_width"]  * scale))
    height   = max(10, int(cfg["output_height"] * scale))
    n_lines  = int(cfg["n_lines"])
    # Divide noise_scale by scale so spatial frequency is resolution-independent
    ns            = float(cfg["noise_scale"]) / max(scale, 0.05)
    a_range       = float(cfg["angle_range"]) * 2.0 * math.pi
    l_median      = float(cfg["length_median"]) * scale
    l_spread      = float(cfg["length_spread"])
    margin        = float(cfg["margin"])
    sw_min        = float(cfg["stroke_width_min"])
    sw_max        = float(cfg["stroke_width_max"])
    a_min         = int(cfg["alpha_min"])
    a_max         = max(a_min + 1, int(cfg["alpha_max"]))
    bg_dark       = bool(cfg["bg_dark"])

    bg = (18,  18,  18)  if bg_dark else (245, 245, 240)
    fg = (220, 220, 215) if bg_dark else (20,  20,  25)

    rng = np.random.default_rng(seed)

    # Starting positions — constrained to the inner rectangle defined by margin
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_lines)
    ys = rng.uniform(y_min, y_max, n_lines)

    # Log-normal length distribution: median controls centre, spread controls the tail.
    # Heavy-tailed — produces a natural mix of fine short marks and sweeping long lines.
    log_lengths = rng.normal(math.log(max(l_median, 1.0)), l_spread, n_lines)
    lengths = np.clip(np.exp(log_lengths), 3.0 * scale, max(width, height) * 1.5)

    # Stroke widths and alphas
    widths = rng.uniform(sw_min, sw_max, n_lines)
    alphas = rng.integers(a_min, a_max, n_lines)

    # Build the angle field (H, W); use a different seed offset for the field
    noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)

    # Sample angle at each line's start position
    xi = np.clip(xs.astype(np.intp), 0, width  - 1)
    yi = np.clip(ys.astype(np.intp), 0, height - 1)
    nv = noise_field[yi, xi]             # ∈ [−1, 1] approximately
    angles = (nv + 1.0) * 0.5 * a_range  # ∈ [0, angle_range]

    # Vectorised line endpoint computation
    halves = lengths * 0.5
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)
    x0s = xs - cos_a * halves
    y0s = ys - sin_a * halves
    x1s = xs + cos_a * halves
    y1s = ys + sin_a * halves

    # Precompute blended colours for every alpha level in [a_min, a_max)
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(a_min, a_max):
        t = a / 255.0
        color_lut[a] = tuple(int(bg[i] + (fg[i] - bg[i]) * t) for i in range(3))  # type: ignore[misc]

    # Draw
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for i in range(n_lines):
        draw.line(
            [(float(x0s[i]), float(y0s[i])), (float(x1s[i]), float(y1s[i]))],
            fill=color_lut.get(int(alphas[i]), fg),
            width=max(1, round(float(widths[i]))),
        )

    return img
