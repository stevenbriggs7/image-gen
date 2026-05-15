"""
Spirograph: hypotrochoid and epitrochoid parametric curves.

A point on a circle rolling inside (hypotrochoid) or outside (epitrochoid)
another fixed circle traces these curves — the same geometry as the classic toy.

Parametrised by:
  R – fixed outer circle "radius" (like gear-tooth count)
  r – rolling inner circle "radius"
  d – pen distance from the center of the rolling circle

Hypotrochoid (inner rolling):
  x(t) = (R−r)·cos(t) + d·cos((R−r)/r · t)
  y(t) = (R−r)·sin(t) − d·sin((R−r)/r · t)

Epitrochoid (outer rolling):
  x(t) = (R+r)·cos(t) − d·cos((R+r)/r · t)
  y(t) = (R+r)·sin(t) − d·sin((R+r)/r · t)

When R and r are integers the curve closes exactly after r/gcd(R,r) full
revolutions, producing symmetric petal patterns with R/gcd(R,r) petals.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _hex_to_rgb, _roughen_path, _taper_width

DEFAULTS: dict = {
    "seed": 42,               # unused — curve is deterministic; kept for API parity
    "output_width": 1200,
    "output_height": 800,
    "R": 7,                   # fixed outer circle radius (integer)
    "r": 3,                   # rolling circle radius (integer)
    "d": 6.0,                 # pen distance from rolling circle center
    "mode": "hypo",           # "hypo" (inner rolling) | "epi" (outer rolling)
    "n_repeats": 1,           # how many times to retrace the closed pattern
    "stroke_width": 1.5,
    "alpha_max": 200,
    "alpha_min": 40,
    "margin": 0.05,
    # Shared keys required by shared UI sections
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
    "stroke_taper": 0.0,
    "stroke_width_var": 0.0,
    "stroke_break_density": 0.0,
    "stroke_roughness": 0.0,
}

_CHUNK = 120  # steps per drawn segment for fade smoothness


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Trace a spirograph curve and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    width     = max(10, int(cfg["output_width"]  * scale))
    height    = max(10, int(cfg["output_height"] * scale))
    R         = float(cfg["R"])
    r         = max(0.5, float(cfg["r"]))
    d         = float(cfg["d"])
    mode      = str(cfg["mode"]).lower()
    n_repeats = max(1, int(cfg["n_repeats"]))
    stroke_w  = max(1, round(float(cfg["stroke_width"])))
    alpha_max = int(cfg["alpha_max"])
    alpha_min = int(cfg["alpha_min"])
    margin    = float(cfg["margin"])
    taper     = float(cfg.get("stroke_taper", 0.0))
    width_var = float(cfg.get("stroke_width_var", 0.0))
    break_p   = float(cfg.get("stroke_break_density", 0.0))
    roughness = float(cfg.get("stroke_roughness", 0.0))

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    # ── 1. Determine closure period ───────────────────────────────────────────
    # When R and r are integers, the curve closes after r/gcd(R,r) outer revolutions.
    R_int = max(2, round(R))
    r_int = max(1, round(r))
    g = math.gcd(R_int, r_int)
    loops = (r_int // g) * n_repeats      # total outer-circle revolutions
    t_max = 2.0 * math.pi * loops

    # ── 2. Compute step count for smooth curves ───────────────────────────────
    # Aim for ≈1.5 canvas pixels between adjacent plotted points.
    canvas_half = min(width, height) * 0.5 * max(1.0 - 2.0 * margin, 0.1)
    n_steps = max(500, int(2.0 * math.pi * canvas_half / 1.5 * loops))
    n_steps = min(n_steps, 400_000)

    # ── 3. Parametric trace ───────────────────────────────────────────────────
    t = np.linspace(0.0, t_max, n_steps + 1, dtype=np.float64)

    if mode == "hypo":
        Rr = R - r
        x = Rr * np.cos(t) + d * np.cos(Rr / r * t)
        y = Rr * np.sin(t) - d * np.sin(Rr / r * t)
    else:  # epi
        Rr = R + r
        x = Rr * np.cos(t) - d * np.cos(Rr / r * t)
        y = Rr * np.sin(t) - d * np.sin(Rr / r * t)

    # ── 4. Scale to canvas ────────────────────────────────────────────────────
    x_span = float(x.max() - x.min()) or 1.0
    y_span = float(y.max() - y.min()) or 1.0
    usable_w = width  * (1.0 - 2.0 * margin)
    usable_h = height * (1.0 - 2.0 * margin)
    fit = min(usable_w / x_span, usable_h / y_span)

    cx, cy = width * 0.5, height * 0.5
    px = cx + (x - (x.max() + x.min()) * 0.5) * fit
    py = cy + (y - (y.max() + y.min()) * 0.5) * fit

    # ── 5. Pre-blended colour LUT ─────────────────────────────────────────────
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(256):
        frac = a / 255.0
        color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * frac) for c in range(3))  # type: ignore[misc]

    # ── 6. Draw with alpha fade along the trace ───────────────────────────────
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    rng  = np.random.default_rng(int(cfg["seed"]))
    path = np.column_stack([px, py])
    if roughness > 0.0:
        path = _roughen_path(path, roughness * stroke_w, rng)

    total = len(px) - 1
    for start in range(0, total, _CHUNK):
        if break_p > 0.0 and rng.random() < break_p:
            continue
        end  = min(start + _CHUNK + 1, total + 1)
        pts  = [(float(path[s, 0]), float(path[s, 1])) for s in range(start, end)]
        if len(pts) < 2:
            continue
        t_frac = start / max(total - 1, 1)
        alpha  = int(alpha_max + (alpha_min - alpha_max) * t_frac)
        eff_w  = stroke_w * (1.0 + width_var * (rng.random() - 0.5) * 2.0) if width_var > 0.0 else stroke_w
        draw.line(pts, fill=color_lut[max(0, min(255, alpha))], width=_taper_width(eff_w, t_frac, taper))

    return img
