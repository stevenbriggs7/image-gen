"""
Isometric cubes: scattered 2.5D cubes with per-face shading and optional warp.

Each cube shows three visible faces whose shade simulates a top-left light source.
An optional global warp (sine wave or radial sphere) displaces each vertex
independently, bending the cube faces as if printed on a curved surface.

Geometry for cube of size s centred at (cx, cy):
  hw  = s * √3/2              (horizontal half-width)
  TOP = (cx,      cy - s)
  TL  = (cx - hw, cy - s/2)
  TR  = (cx + hw, cy - s/2)
  MID = (cx,      cy)          ← inner vertex where three faces meet
  BL  = (cx - hw, cy + s/2)
  BR  = (cx + hw, cy + s/2)
  BOT = (cx,      cy + s)

  Top face:   TOP, TR, MID, TL
  Right face: TR,  BR, BOT, MID
  Left face:  TL,  MID, BOT, BL
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _fractal_noise_field, _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_cubes": 200,
    "size_median": 40.0,
    "size_spread": 0.5,
    "noise_scale": 0.004,
    "noise_influence": 0.4,
    "filled": True,
    "stroke_width": 1.0,
    "alpha_min": 60,
    "alpha_max": 200,
    "shade_contrast": 0.8,   # 0 = all faces same; 1 = max light/shadow difference
    "warp_type": "none",     # "none" | "wave" | "sphere"
    "warp_amplitude": 0.0,   # displacement as fraction of cube size
    "warp_scale": 0.005,     # wave spatial frequency or sphere falloff rate
    "margin": 0.05,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
}


def _warp_pt(
    vx: float, vy: float,
    warp_type: str, amp: float, scale: float,
    canvas_cx: float, canvas_cy: float,
) -> tuple[float, float]:
    """Displace a single vertex according to the warp field."""
    if warp_type == "wave":
        return (vx + amp * math.sin(scale * vx),
                vy + amp * math.cos(scale * vy))
    elif warp_type == "sphere":
        dx, dy = vx - canvas_cx, vy - canvas_cy
        r = math.sqrt(dx * dx + dy * dy) + 1e-9
        factor = amp / (1.0 + r * scale)
        return vx + dx * factor, vy + dy * factor
    return vx, vy


def _cube_verts(
    cx: float, cy: float, s: float,
    warp_type: str, amp: float, scale: float,
    canvas_cx: float, canvas_cy: float,
) -> tuple:
    """Return the 7 warped vertices of an isometric cube."""
    hw = s * math.sqrt(3) / 2

    raw = [
        (cx,      cy - s),       # TOP
        (cx - hw, cy - s / 2),   # TL
        (cx + hw, cy - s / 2),   # TR
        (cx,      cy),            # MID
        (cx - hw, cy + s / 2),   # BL
        (cx + hw, cy + s / 2),   # BR
        (cx,      cy + s),        # BOT
    ]

    if warp_type == "none" or amp == 0.0:
        return tuple(raw)

    return tuple(
        _warp_pt(vx, vy, warp_type, amp, scale, canvas_cx, canvas_cy)
        for vx, vy in raw
    )


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render scattered isometric cubes and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_cubes         = int(cfg["n_cubes"])
    l_median        = float(cfg["size_median"]) * scale
    l_spread        = float(cfg["size_spread"])
    ns              = float(cfg["noise_scale"]) / max(scale, 0.05)
    noise_inf       = float(cfg["noise_influence"])
    filled          = bool(cfg["filled"])
    stroke_w        = max(1, round(float(cfg["stroke_width"])))
    a_min           = int(cfg["alpha_min"])
    a_max           = max(a_min + 1, int(cfg["alpha_max"]))
    shade_contrast  = float(cfg["shade_contrast"])
    warp_type       = str(cfg["warp_type"]).lower()
    warp_amp_frac   = float(cfg["warp_amplitude"])
    warp_scale      = float(cfg["warp_scale"])
    margin          = float(cfg["margin"])
    gravity         = float(cfg["gravity"])
    gravity_falloff = float(cfg["gravity_falloff"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng = np.random.default_rng(seed)

    # ── 1. Scatter cube centres ───────────────────────────────────────────────
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_cubes)
    ys = rng.uniform(y_min, y_max, n_cubes)

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

    # ── 2. Sizes (log-normal, noise-modulated) ────────────────────────────────
    log_sizes = rng.normal(math.log(max(l_median, 1.0)), l_spread, n_cubes)
    sizes = np.clip(np.exp(log_sizes), 2.0 * scale, min(width, height) * 0.4)

    alphas = rng.integers(a_min, a_max, n_cubes).astype(float)
    if noise_inf > 0.0:
        noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)
        xi = np.clip(xs.astype(np.intp), 0, width  - 1)
        yi = np.clip(ys.astype(np.intp), 0, height - 1)
        nv = noise_field[yi, xi]
        sizes  = sizes * np.clip(1.0 + nv * noise_inf, 0.15, 2.5)
        alphas = np.clip(alphas + nv * noise_inf * (a_max - a_min), a_min, a_max - 1)

    # ── 3. Alphas + painter sort (top of canvas first) ────────────────────────
    alphas = alphas.astype(int)
    order  = np.argsort(ys)   # ascending cy → draw topmost (furthest) first

    # ── 4. Shade factors for 3 faces ─────────────────────────────────────────
    shade_top   = 1.0
    shade_right = 1.0 - 0.35 * shade_contrast
    shade_left  = 1.0 - 0.65 * shade_contrast

    # ── 5. Pre-blended colour LUT ─────────────────────────────────────────────
    color_lut: dict[int, tuple[int, int, int]] = {}
    for a in range(256):
        frac = a / 255.0
        color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * frac) for c in range(3))  # type: ignore[misc]

    def face_color(alpha: int, shade: float) -> tuple[int, int, int]:
        a = max(0, min(255, round(alpha * shade)))
        return color_lut[a]

    # ── 6. Draw ───────────────────────────────────────────────────────────────
    canvas_cx, canvas_cy = width * 0.5, height * 0.5
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for idx in order:
        cx  = float(xs[idx])
        cy  = float(ys[idx])
        s   = float(sizes[idx])
        alp = int(alphas[idx])
        amp = warp_amp_frac * s   # warp in pixels, scaled to cube size

        top, tl, tr, mid, bl, br, bot = _cube_verts(
            cx, cy, s, warp_type, amp, warp_scale, canvas_cx, canvas_cy,
        )

        faces = [
            ([top, tr, mid, tl], shade_top),
            ([tr, br, bot, mid], shade_right),
            ([tl, mid, bot, bl], shade_left),
        ]

        if filled:
            for pts, shade in faces:
                fill    = face_color(alp, shade)
                outline = face_color(max(0, alp - 40), shade * 0.7) if stroke_w > 0 else None
                draw.polygon(pts, fill=fill, outline=outline, width=stroke_w)
        else:
            for pts, shade in faces:
                outline = face_color(alp, shade)
                draw.polygon(pts, fill=None, outline=outline, width=stroke_w)

    return img
