"""
Generative wave lines: parallel lines whose centres ride a sine wave.

Think of a comb of vertical lines whose spine bends into a wave — the wave
determines where each line's centre sits, not the angle of the lines themselves.

Mathematical principles:
  - Line centres are placed directly ON the wave curve (u sampled along the
    wave axis, v set to wave_v = A·sin(...)).
  - All lines point in the same fixed direction — perpendicular to the wave's
    travel direction (vertical for a horizontal wave, etc.).
  - `line_scatter` adds optional Gaussian spread of centres off the wave,
    softening the mechanical comb look without changing the average position.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _hex_to_rgb, _invert_segment, _roughen_path

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
    # Line placement
    "line_scatter": 0.0,      # spread of centres off the wave (fraction of canvas height)
    "angle_spread": 0.0,      # std-dev radians of jitter around the perpendicular
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
    "stroke_taper": 0.0,
    "stroke_width_var": 0.0,
    "stroke_break_density": 0.0,
    "stroke_roughness": 0.0,
}


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render wave-spine lines and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_lines         = int(cfg["n_lines"])
    wave_amp        = float(cfg["wave_amplitude"]) * height
    wave_freq       = float(cfg["wave_frequency"])
    wave_phase      = float(cfg["wave_phase"]) * 2.0 * math.pi
    theta           = math.radians(float(cfg["wave_angle"]))
    line_scatter    = float(cfg["line_scatter"]) * height
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
    break_p  = float(cfg.get("stroke_break_density", 0.0))
    roughness = float(cfg.get("stroke_roughness", 0.0))

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng  = np.random.default_rng(seed)
    span = float(width)   # frequency is "cycles per canvas width"
    cx, cy = width * 0.5, height * 0.5
    cos_t  = math.cos(theta)
    sin_t  = math.sin(theta)

    # ── 1. Sample u positions along the wave axis ─────────────────────────────
    # u is the coordinate along the wave's travel direction (like x for a
    # horizontal wave). Centres are spread uniformly across the canvas.
    u_half = width * (0.5 - margin)
    us = rng.uniform(-u_half, u_half, n_lines)

    # ── 2. Place centres on the wave curve ───────────────────────────────────
    phase_arg = 2.0 * math.pi * wave_freq * us / span + wave_phase
    wave_vs   = wave_amp * np.sin(phase_arg)

    # Optional scatter: spread centres off the wave perpendicular to travel
    if line_scatter > 0.0:
        wave_vs = wave_vs + rng.normal(0.0, line_scatter, n_lines)

    # ── 3. Back-transform to canvas coordinates ───────────────────────────────
    xs = cx + us * cos_t - wave_vs * sin_t
    ys = cy + us * sin_t + wave_vs * cos_t

    # ── 4. Optional canvas-centre gravity (shared with other generators) ───────
    if gravity > 0.0:
        dx, dy = cx - xs, cy - ys
        if gravity_falloff > 0.0:
            canvas_r = 0.5 * math.sqrt(width ** 2 + height ** 2)
            d = np.sqrt(dx ** 2 + dy ** 2)
            weight = np.exp(-gravity_falloff * 5.0 * d / (canvas_r + 1e-9))
        else:
            weight = 1.0
        eff = math.sqrt(gravity)
        xs = xs + dx * eff * weight
        ys = ys + dy * eff * weight

    # ── 5. Line angles: perpendicular to wave travel, with optional jitter ───────
    angles = np.full(n_lines, theta + math.pi / 2)
    if angle_spread > 0.0:
        angles = angles + rng.normal(0.0, angle_spread, n_lines)

    # ── 6. Log-normal length distribution ─────────────────────────────────────
    log_lengths = rng.normal(math.log(max(l_median, 1.0)), l_spread, n_lines)
    lengths     = np.clip(np.exp(log_lengths), 3.0 * scale, max(width, height) * 1.5)

    # ── 7. Stroke widths and alphas ───────────────────────────────────────────
    widths = rng.uniform(sw_min, sw_max, n_lines)
    alphas = rng.integers(a_min, a_max, n_lines)

    # ── 8. Vectorised endpoint computation ────────────────────────────────────
    halves = lengths * 0.5
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)
    x0s = xs - cos_a * halves
    y0s = ys - sin_a * halves
    x1s = xs + cos_a * halves
    y1s = ys + sin_a * halves

    # ── 9. Draw ───────────────────────────────────────────────────────────────
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
            if break_p > 0.0 and rng.random() < break_p:
                continue
            x0, y0 = float(x0s[i]), float(y0s[i])
            x1, y1 = float(x1s[i]), float(y1s[i])
            w = max(1, round(float(widths[i])))
            if roughness > 0.0:
                dx, dy = x1 - x0, y1 - y0
                ln = math.sqrt(dx * dx + dy * dy) + 1e-9
                off = rng.normal(0.0, roughness * w)
                line_pts: list = [(x0, y0),
                                  ((x0 + x1) * 0.5 - (dy / ln) * off,
                                   (y0 + y1) * 0.5 + (dx / ln) * off),
                                  (x1, y1)]
            else:
                line_pts = [(x0, y0), (x1, y1)]
            draw.line(line_pts, fill=color_lut.get(int(alphas[i]), fg), width=w)

    return img
