"""
Isometric grid: cubes tiling the entire canvas, every edge shared with a neighbour.

Grid coordinates (i, j) map to screen via:
  cx(i, j)    = cx0 + (i − j) · hw        hw = s · √3/2
  cy(i, j, k) = cy0 + (i + j) · hs − k·s  hs = s / 2

Draw order: iterate depth d = i+j from 0 (back) to d_max (front), then
stacking height k from 0 (floor) to h−1 (top). This gives correct
painter's-algorithm occlusion for any height configuration.

Height styles:
  flat    — uniform height 1 (classic isometric floor)
  terrain — smooth fractal noise → organic hills
  towers  — independent random integer height per column
  wave    — sinusoidal ripple pattern
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _fractal_noise_field, _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "cube_size": 40,             # s in pixels (half the total screen height of one cube)
    "max_height": 1,             # maximum stacked cubes per column
    "height_style": "flat",      # "flat" | "terrain" | "towers" | "wave"
    "height_noise_scale": 0.005, # spatial frequency of height variation
    "shade_contrast": 0.75,      # 0 = all faces same tone; 1 = strong light/shadow
    "noise_scale": 0.006,        # spatial frequency of per-cube tone variation
    "noise_influence": 0.25,     # strength of tone variation
    "filled": True,
    "stroke_width": 1.0,
    "alpha_min": 140,
    "alpha_max": 220,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
}


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render a fully tiled isometric cube grid and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    seed         = int(cfg["seed"])
    width        = max(10, int(cfg["output_width"]  * scale))
    height       = max(10, int(cfg["output_height"] * scale))
    s            = max(2.0, float(cfg["cube_size"]) * scale)
    hw           = s * math.sqrt(3) / 2   # horizontal half-width
    hs           = s / 2                  # vertical half-step per depth level
    max_h        = max(1, int(cfg["max_height"]))
    h_style      = str(cfg["height_style"]).lower()
    h_ns         = float(cfg["height_noise_scale"]) / max(scale, 0.05)
    shade_ctr    = float(cfg["shade_contrast"])
    ns           = float(cfg["noise_scale"]) / max(scale, 0.05)
    noise_inf    = float(cfg["noise_influence"])
    filled       = bool(cfg["filled"])
    stroke_w     = max(1, round(float(cfg["stroke_width"])))
    a_min        = int(cfg["alpha_min"])
    a_max        = max(a_min + 1, int(cfg["alpha_max"]))

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng = np.random.default_rng(seed)

    # ── Grid origin ───────────────────────────────────────────────────────────
    # cy0 = s means the very first row's TOP vertex lands at y = 0.
    cx0 = width * 0.5
    cy0 = float(s)

    # Extent: how many columns (h_half) and depth levels (d_max) to cover canvas
    h_half = int(math.ceil((width * 0.5 + s * 2.0) / hw)) + 1
    d_max  = int(math.ceil(height / hs)) + max_h * 2 + 4

    # ── Noise fields ──────────────────────────────────────────────────────────
    tone_field   = None
    height_field = None
    if noise_inf > 0.0:
        tone_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)
    if max_h > 1 and h_style == "terrain":
        height_field = _fractal_noise_field(width, height, h_ns, seed=seed + 55555)

    # ── Pre-generate tower heights (deterministic per column) ─────────────────
    tower_map: dict[tuple[int, int], int] = {}
    if max_h > 1 and h_style == "towers":
        for d in range(d_max + 1):
            i_lo = max(0, math.floor((d - h_half) / 2))
            i_hi = min(d, math.ceil((d + h_half) / 2))
            for i in range(i_lo, i_hi + 1):
                j = d - i
                if (i, j) not in tower_map:
                    tower_map[(i, j)] = int(rng.integers(1, max_h + 1))

    # ── Height lookup ─────────────────────────────────────────────────────────
    def get_height(i: int, j: int, cx_f: float, cy_ground: float) -> int:
        if max_h == 1 or h_style == "flat":
            return 1
        if h_style == "terrain" and height_field is not None:
            xi = int(max(0, min(width  - 1, cx_f)))
            yi = int(max(0, min(height - 1, cy_ground)))
            nv = float(height_field[yi, xi])   # ∈ [-1, 1]
            return 1 + int(round((nv + 1) * 0.5 * (max_h - 1)))
        if h_style == "towers":
            return tower_map.get((i, j), 1)
        if h_style == "wave":
            nv = math.sin(h_ns * 150.0 * (i + j)) * 0.5 + 0.5
            return 1 + int(round(nv * (max_h - 1)))
        return 1

    # ── Shade factors (top light, right mid, left dark) ───────────────────────
    shade_top   = 1.0
    shade_right = 1.0 - 0.35 * shade_ctr
    shade_left  = 1.0 - 0.65 * shade_ctr

    # ── Colour LUT ────────────────────────────────────────────────────────────
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(256):
        t = a / 255.0
        color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * t) for c in range(3))  # type: ignore[misc]

    def face_color(alpha: int, shade: float) -> tuple[int, int, int]:
        return color_lut[max(0, min(255, round(alpha * shade)))]

    base_alpha = (a_min + a_max) // 2

    def get_alpha(cx_f: float, cy_f: float) -> int:
        if tone_field is None:
            return base_alpha
        xi = int(max(0, min(width  - 1, cx_f)))
        yi = int(max(0, min(height - 1, cy_f)))
        nv = float(tone_field[yi, xi])
        return int(max(a_min, min(a_max, base_alpha + nv * noise_inf * (a_max - a_min) * 0.5)))

    # ── Draw — back-to-front, bottom-to-top ───────────────────────────────────
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for d in range(d_max + 1):
        cy_ground = cy0 + d * hs
        # Skip this depth row if all cubes (even the tallest) are below canvas
        if cy_ground - max_h * s > height + s:
            continue

        i_lo = max(0, math.floor((d - h_half) / 2))
        i_hi = min(d,  math.ceil((d + h_half) / 2))

        for i in range(i_lo, i_hi + 1):
            j  = d - i
            cx = cx0 + (i - j) * hw

            if cx < -s * 2.0 or cx > width + s * 2.0:
                continue

            h_ij = get_height(i, j, cx, cy_ground)

            for k in range(h_ij):
                cy = cy_ground - k * s

                # Cheap visibility cull (TOP vertex above bottom, BOT below top)
                if cy - s > height + s or cy + s < -s:
                    continue

                alp = get_alpha(cx, cy)

                # 7 vertices
                top = (cx,      cy - s)
                tl  = (cx - hw, cy - hs)
                tr  = (cx + hw, cy - hs)
                mid = (cx,      cy)
                bl  = (cx - hw, cy + hs)
                br  = (cx + hw, cy + hs)
                bot = (cx,      cy + s)

                faces = [
                    ([top, tr, mid, tl], shade_top),
                    ([tr, br, bot, mid], shade_right),
                    ([tl, mid, bot, bl], shade_left),
                ]

                if filled:
                    for pts, shade in faces:
                        fill    = face_color(alp, shade)
                        outline = face_color(max(0, alp - 30), shade * 0.7) if stroke_w > 0 else None
                        draw.polygon(pts, fill=fill, outline=outline, width=stroke_w)
                else:
                    for pts, shade in faces:
                        draw.polygon(pts, fill=None,
                                     outline=face_color(alp, shade), width=stroke_w)

    return img
