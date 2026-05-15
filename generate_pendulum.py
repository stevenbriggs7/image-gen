"""
Pendulum pour: a paint pot on a rope swung in a circle or ellipse over a canvas.

The pot traces a circular/elliptical spiral that tightens as the swing decays.
`aspect` squashes the circle into an ellipse; `tilt` rotates the ellipse axes.
Paint fades as the pot empties.

Equation:
  x_raw(t) = A · cos(t) · exp(−d·t)
  y_raw(t) = A · aspect · sin(t) · exp(−d·t)
  [x, y] = rotate [x_raw, y_raw] by tilt degrees
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    # Pendulum shape
    "aspect": 1.0,       # y/x amplitude ratio: 1.0 = circle, <1 = squashed
    "tilt": 0.0,         # rotation of ellipse axes in degrees
    "damping": 0.0003,   # amplitude decay rate — controls how tight the spiral is
    "n_steps": 60000,    # simulation steps (more = more loops before centre)
    "amplitude": 0.88,   # initial swing as fraction of min(width, height)/2
    # Paint
    "flow_rate": 0.7,    # opacity fade rate (0=constant ink, 2=fast fade)
    "stroke_width": 2.0,
    "alpha_max": 200,
    "alpha_min": 8,
    # Shared (read by shared UI sections)
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    # Colours
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
}

_CHUNK = 200  # steps per polyline batch


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Simulate a damped circular/elliptical pendulum and render the paint trail.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    width      = max(10, int(cfg["output_width"]  * scale))
    height     = max(10, int(cfg["output_height"] * scale))
    aspect     = float(cfg["aspect"])
    tilt_rad   = math.radians(float(cfg["tilt"]))
    damping    = float(cfg["damping"])
    n_steps    = int(cfg["n_steps"])
    amplitude  = float(cfg["amplitude"])
    flow_rate  = float(cfg["flow_rate"])
    stroke_w   = max(1, round(float(cfg["stroke_width"])))
    alpha_max  = int(cfg["alpha_max"])
    alpha_min  = int(cfg["alpha_min"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    # ── 1. Time array ─────────────────────────────────────────────────────────
    dt = 0.05
    t  = np.arange(n_steps, dtype=np.float64) * dt

    # ── 2. Elliptical path with decay ─────────────────────────────────────────
    cx, cy  = width * 0.5, height * 0.5
    R       = min(width, height) * 0.5 * amplitude
    decay   = np.exp(-damping * t)

    x_raw = R * np.cos(t) * decay
    y_raw = R * aspect * np.sin(t) * decay

    # ── 3. Tilt rotation ──────────────────────────────────────────────────────
    cos_tilt = math.cos(tilt_rad)
    sin_tilt = math.sin(tilt_rad)
    xs = cx + x_raw * cos_tilt - y_raw * sin_tilt
    ys = cy + x_raw * sin_tilt + y_raw * cos_tilt

    # ── 4. Per-step opacity (pot emptying) ────────────────────────────────────
    t_norm = t / max(t[-1], 1.0)
    alphas = alpha_min + (alpha_max - alpha_min) * np.exp(-flow_rate * t_norm)
    alphas = np.clip(alphas, 0, 255).astype(np.int32)

    # ── 5. Pre-blended colour LUT ─────────────────────────────────────────────
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(256):
        frac = a / 255.0
        color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * frac) for c in range(3))  # type: ignore[misc]

    # ── 6. Draw in batches ────────────────────────────────────────────────────
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for start in range(0, n_steps - 1, _CHUNK):
        end  = min(start + _CHUNK + 1, n_steps)
        pts  = [(float(xs[i]), float(ys[i])) for i in range(start, end)]
        if len(pts) < 2:
            continue
        a     = int(np.median(alphas[start:end]))
        color = color_lut.get(a, fg)
        draw.line(pts, fill=color, width=stroke_w)

    return img
