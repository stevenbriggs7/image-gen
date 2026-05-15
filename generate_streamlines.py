"""
Flow streamlines: long continuous curves tracing a Perlin noise flow field.

Like the Lines generator but each mark travels far across the canvas instead
of stopping after a short stroke. The result resembles ink dropped in water —
river-like paths that layer and cross organically.

Each streamline fades from alpha_max at its start to alpha_min at its end,
giving a natural ink-thinning effect as the flow carries the mark.

Reuses _fractal_noise_field and _hex_to_rgb from generate.py.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _fractal_noise_field, _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_lines": 80,        # number of streamlines
    "n_steps": 600,       # steps traced per streamline
    "step_size": 5.0,     # pixels moved per step
    "noise_scale": 0.004,
    "angle_range": 1.0,   # fraction of 2π covered by the field
    "stroke_width": 1.5,
    "alpha_max": 160,     # opacity at the start of each streamline
    "alpha_min": 15,      # opacity at the end (ink thinning)
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
}

_CHUNK = 40  # steps per drawn segment (controls fade smoothness)


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Trace flow streamlines through a fractal noise field and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_lines         = int(cfg["n_lines"])
    n_steps         = int(cfg["n_steps"])
    step_size       = float(cfg["step_size"]) * scale
    ns              = float(cfg["noise_scale"]) / max(scale, 0.05)
    a_range         = float(cfg["angle_range"]) * 2.0 * math.pi
    stroke_w        = max(1, round(float(cfg["stroke_width"])))
    alpha_max       = int(cfg["alpha_max"])
    alpha_min       = int(cfg["alpha_min"])
    margin          = float(cfg["margin"])
    gravity         = float(cfg["gravity"])
    gravity_falloff = float(cfg["gravity_falloff"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng = np.random.default_rng(seed)

    # ── 1. Starting positions ─────────────────────────────────────────────────
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_lines)
    ys = rng.uniform(y_min, y_max, n_lines)

    if gravity > 0.0:
        cx0, cy0 = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        dx, dy = cx0 - xs, cy0 - ys
        if gravity_falloff > 0.0:
            canvas_r = 0.5 * math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            d = np.sqrt(dx ** 2 + dy ** 2)
            weight = np.exp(-gravity_falloff * 5.0 * d / (canvas_r + 1e-9))
        else:
            weight = 1.0
        xs = xs + dx * math.sqrt(gravity) * weight
        ys = ys + dy * math.sqrt(gravity) * weight

    # ── 2. Build noise field ──────────────────────────────────────────────────
    noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)

    # ── 3. Trace all streamlines in parallel (vectorised per step) ───────────
    # all_pts: (n_lines, n_steps+1, 2)
    all_pts = np.empty((n_lines, n_steps + 1, 2), dtype=np.float32)
    all_pts[:, 0, 0] = xs
    all_pts[:, 0, 1] = ys

    x_curr = xs.copy()
    y_curr = ys.copy()

    xi = np.clip(x_curr.astype(np.intp), 0, width  - 1)
    yi = np.clip(y_curr.astype(np.intp), 0, height - 1)
    cur_angles = (noise_field[yi, xi] + 1.0) * 0.5 * a_range

    for s in range(n_steps):
        x_curr = x_curr + np.cos(cur_angles) * step_size
        y_curr = y_curr + np.sin(cur_angles) * step_size
        all_pts[:, s + 1, 0] = x_curr
        all_pts[:, s + 1, 1] = y_curr
        if s < n_steps - 1:
            xi = np.clip(x_curr.astype(np.intp), 0, width  - 1)
            yi = np.clip(y_curr.astype(np.intp), 0, height - 1)
            cur_angles = (noise_field[yi, xi] + 1.0) * 0.5 * a_range

    # ── 4. Pre-blended colour LUT ─────────────────────────────────────────────
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(256):
        frac = a / 255.0
        color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * frac) for c in range(3))  # type: ignore[misc]

    # ── 5. Draw each streamline in fading chunks ──────────────────────────────
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for i in range(n_lines):
        for start in range(0, n_steps, _CHUNK):
            end = min(start + _CHUNK + 1, n_steps + 1)
            pts = [(float(all_pts[i, s, 0]), float(all_pts[i, s, 1]))
                   for s in range(start, end)]
            if len(pts) < 2:
                continue
            t_frac = start / max(n_steps - 1, 1)
            alpha  = int(alpha_max + (alpha_min - alpha_max) * t_frac)
            alpha  = max(0, min(255, alpha))
            draw.line(pts, fill=color_lut.get(alpha, fg), width=stroke_w)

    return img
