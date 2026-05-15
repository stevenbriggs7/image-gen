"""
Generative wave lines: strokes oriented perpendicular to an invisible sine wave.

Mathematical principles:
  - A sine wave runs across the canvas at a configurable angle.
  - Lines are scattered across the canvas and oriented perpendicular to the
    wave's local tangent at their position, so the wave acts as a direction field.
  - Wave gravity pulls mark positions toward the wave curve itself, creating
    natural density clustering along the wave.
  - Perpendicular strength blends between random orientation and true
    perpendicular, so distant marks can drift while nearby marks align tightly.
  - Angular spread adds noise around the perpendicular for organic variation.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _hex_to_rgb, _invert_segment

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_lines": 1200,
    # Wave shape
    "wave_amplitude": 0.20,   # fraction of canvas height
    "wave_frequency": 2.0,    # complete cycles across canvas width
    "wave_phase": 0.0,        # 0–1, maps to 0–2π phase offset
    "wave_angle": 0.0,        # degrees; 0 = horizontal wave, 90 = vertical
    # Wave influence on line placement
    "wave_gravity": 0.4,      # 0 = uniform scatter, 1 = all lines on the wave
    # Wave influence on line orientation
    "perp_strength": 0.75,    # 0 = random angles, 1 = perfectly perpendicular
    "angle_spread": 0.4,      # std-dev radians of jitter around perpendicular
    # Line style
    "length_median": 45.0,
    "length_spread": 0.6,
    "stroke_width_min": 1.0,
    "stroke_width_max": 2.5,
    "alpha_min": 50,
    "alpha_max": 180,
    # Shared
    "margin": 0.05,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
    "invert_overlap": False,
}


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render wave-guided lines and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews)
    while preserving the spatial character of the wave.
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_lines         = int(cfg["n_lines"])
    wave_amp        = float(cfg["wave_amplitude"]) * height   # px at current scale
    wave_freq       = float(cfg["wave_frequency"])
    wave_phase      = float(cfg["wave_phase"]) * 2.0 * math.pi
    theta           = math.radians(float(cfg["wave_angle"]))
    wave_gravity    = float(cfg["wave_gravity"])
    perp_strength   = float(cfg["perp_strength"])
    angle_spread    = float(cfg["angle_spread"])
    l_median        = float(cfg["length_median"]) * scale
    l_spread        = float(cfg["length_spread"])
    margin          = float(cfg["margin"])
    gravity         = float(cfg["gravity"])
    gravity_falloff = float(cfg["gravity_falloff"])
    sw_min          = float(cfg["stroke_width_min"])
    sw_max          = float(cfg["stroke_width_max"])
    a_min           = int(cfg["alpha_min"])
    a_max           = max(a_min + 1, int(cfg["alpha_max"]))
    invert_overlap  = bool(cfg["invert_overlap"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng  = np.random.default_rng(seed)
    span = float(width)   # frequency is "cycles per canvas width"

    # ── 1. Sample positions within margin ────────────────────────────────────
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_lines)
    ys = rng.uniform(y_min, y_max, n_lines)

    # ── 2. Canvas-centre gravity (shared with other generators) ───────────────
    if gravity > 0.0:
        cx0, cy0 = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        dx, dy = cx0 - xs, cy0 - ys
        if gravity_falloff > 0.0:
            canvas_r = 0.5 * math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            d = np.sqrt(dx ** 2 + dy ** 2)
            weight = np.exp(-gravity_falloff * 5.0 * d / (canvas_r + 1e-9))
        else:
            weight = 1.0
        eff = math.sqrt(gravity)
        xs = xs + dx * eff * weight
        ys = ys + dy * eff * weight

    # ── 3. Transform to wave-aligned coordinates ──────────────────────────────
    cx, cy = width * 0.5, height * 0.5
    cos_t  = math.cos(theta)
    sin_t  = math.sin(theta)

    dx_c = xs - cx
    dy_c = ys - cy
    u = dx_c * cos_t + dy_c * sin_t    # along-wave axis
    v = -dx_c * sin_t + dy_c * cos_t  # across-wave axis

    # ── 4. Wave position and signed distance ──────────────────────────────────
    phase_arg = 2.0 * math.pi * wave_freq * u / span + wave_phase
    wave_v    = wave_amp * np.sin(phase_arg)
    dist      = v - wave_v

    # ── 5. Wave gravity: pull positions toward the wave curve ─────────────────
    if wave_gravity > 0.0:
        v_new = v - dist * math.sqrt(wave_gravity)
    else:
        v_new = v

    # Convert back to canvas coordinates
    xs = cx + u * cos_t - v_new * sin_t
    ys = cy + u * sin_t + v_new * cos_t

    # ── 6. Perpendicular angle in canvas space ────────────────────────────────
    # Lines point perpendicular to the wave's travel direction (not the local slope).
    # For wave_angle=0° (horizontal), lines are vertical (π/2). Scalar, not per-point.
    perp_angle = theta + math.pi / 2

    # ── 7. Blend with random angles and add spread ────────────────────────────
    rand_angles = rng.uniform(0.0, 2.0 * math.pi, n_lines)
    angles      = perp_angle + (rand_angles - perp_angle) * (1.0 - perp_strength)
    angles     += rng.normal(0.0, angle_spread, n_lines)

    # ── 8. Log-normal length distribution ────────────────────────────────────
    log_lengths = rng.normal(math.log(max(l_median, 1.0)), l_spread, n_lines)
    lengths     = np.clip(np.exp(log_lengths), 3.0 * scale, max(width, height) * 1.5)

    # ── 9. Stroke widths and alphas ───────────────────────────────────────────
    widths = rng.uniform(sw_min, sw_max, n_lines)
    alphas = rng.integers(a_min, a_max, n_lines)

    # ── 10. Vectorised endpoint computation ───────────────────────────────────
    halves = lengths * 0.5
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)
    x0s = xs - cos_a * halves
    y0s = ys - sin_a * halves
    x1s = xs + cos_a * halves
    y1s = ys + sin_a * halves

    # ── 11. Draw ──────────────────────────────────────────────────────────────
    img = Image.new("RGB", (width, height), bg)

    if invert_overlap:
        canvas_arr = np.array(img, dtype=np.float32)
        for i in range(n_lines):
            _invert_segment(
                canvas_arr,
                float(x0s[i]), float(y0s[i]), float(x1s[i]), float(y1s[i]),
                max(1.0, float(widths[i])), float(alphas[i]) / 255.0,
                width, height,
            )
        img = Image.fromarray(canvas_arr.astype(np.uint8), "RGB")
    else:
        color_lut: dict[int, tuple[int, int, int]] = {}
        for a in range(a_min, a_max):
            t = a / 255.0
            color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * t) for c in range(3))  # type: ignore[misc]
        draw = ImageDraw.Draw(img)
        for i in range(n_lines):
            draw.line(
                [(float(x0s[i]), float(y0s[i])), (float(x1s[i]), float(y1s[i]))],
                fill=color_lut.get(int(alphas[i]), fg),
                width=max(1, round(float(widths[i]))),
            )

    return img
