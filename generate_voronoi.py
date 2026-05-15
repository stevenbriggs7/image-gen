"""
Voronoi cells: partition the canvas into nearest-neighbour regions.

Each seed point owns the area of canvas closer to it than to any other seed.
A bounding-mirror trick forces all cells to be finite: every seed is reflected
across all four canvas edges before computing the diagram, so no cell extends
to infinity and no clipping is needed.

Cell opacity is drawn from a fractal noise field for organic clustering —
dense cells clump together, sparse ones open up, giving a natural cracked-earth
or stained-glass texture.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi

from generate import _fractal_noise_field, _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_cells": 120,
    "noise_scale": 0.004,
    "noise_influence": 0.65,
    "filled": True,
    "stroke_width": 1.5,
    "alpha_min": 20,
    "alpha_max": 200,
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
}


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Scatter seed points, compute their Voronoi diagram, and render each cell.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_cells         = max(3, int(cfg["n_cells"]))
    ns              = float(cfg["noise_scale"]) / max(scale, 0.05)
    noise_inf       = float(cfg["noise_influence"])
    filled          = bool(cfg["filled"])
    stroke_w        = max(1, round(float(cfg["stroke_width"])))
    a_min           = int(cfg["alpha_min"])
    a_max           = max(a_min + 1, int(cfg["alpha_max"]))
    margin          = float(cfg["margin"])
    gravity         = float(cfg["gravity"])
    gravity_falloff = float(cfg["gravity_falloff"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng = np.random.default_rng(seed)

    # ── 1. Seed positions ─────────────────────────────────────────────────────
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_cells)
    ys = rng.uniform(y_min, y_max, n_cells)

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

    # Keep points strictly inside canvas so mirrors are distinct
    xs = np.clip(xs, 1.0, width  - 1.0)
    ys = np.clip(ys, 1.0, height - 1.0)

    # ── 2. Noise field for alpha modulation ───────────────────────────────────
    if noise_inf > 0.0:
        noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)

    # ── 3. Bounded Voronoi via mirroring ──────────────────────────────────────
    # Reflecting seeds across all 4 edges forces every original cell to be a
    # finite polygon bounded by the canvas rectangle.
    pts = np.column_stack([xs, ys])
    n   = len(pts)
    mirror_pts = np.vstack([
        pts,
        np.column_stack([-pts[:, 0],          pts[:, 1]]),           # left
        np.column_stack([2 * width - pts[:, 0], pts[:, 1]]),          # right
        np.column_stack([pts[:, 0],            -pts[:, 1]]),           # top
        np.column_stack([pts[:, 0],  2 * height - pts[:, 1]]),        # bottom
    ])
    vor = Voronoi(mirror_pts)

    # ── 4. Per-cell alpha ──────────────────────────────────────────────────────
    alphas = rng.integers(a_min, a_max, n).astype(float)
    if noise_inf > 0.0:
        xi = np.clip(xs.astype(np.intp), 0, width  - 1)
        yi = np.clip(ys.astype(np.intp), 0, height - 1)
        nv = noise_field[yi, xi]
        alphas = alphas + nv * noise_inf * (a_max - a_min) * 0.5
        alphas = np.clip(alphas, a_min, a_max - 1).astype(int)
    else:
        alphas = alphas.astype(int)

    # Colour lookup: alpha → pre-blended RGB
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(a_min, a_max):
        frac = a / 255.0
        color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * frac) for c in range(3))  # type: ignore[misc]

    # ── 5. Draw each cell ─────────────────────────────────────────────────────
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    outline_color = fg  # always use fg for cell edges

    for i in range(n):
        region_idx = vor.point_region[i]
        region     = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        verts = vor.vertices[region]
        poly  = [(float(v[0]), float(v[1])) for v in verts]
        if len(poly) < 3:
            continue

        color = color_lut.get(int(alphas[i]), fg)

        if filled:
            draw.polygon(poly, fill=color, outline=outline_color, width=stroke_w)
        else:
            draw.polygon(poly, fill=None, outline=color, width=stroke_w)

    return img
