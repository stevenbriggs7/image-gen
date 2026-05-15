"""
Generative shapes: scattered circles, triangles and squares.

Mathematical principles:
  - Same fractal value noise field as the circles generator modulates size.
  - All three shape types share a common log-normal size distribution.
  - Shapes are drawn largest-first (painter's algorithm) so small marks sit on top.
  - Triangles and squares use circumradius so all shapes are visually comparable
    at the same size setting.
  - Rotation randomises each polygon's orientation independently.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _fractal_noise_field, _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_circles": 200,
    "n_triangles": 200,
    "n_squares": 200,
    "size_median": 20.0,     # median circumradius in px at full resolution
    "size_spread": 0.65,     # log-sigma of size distribution
    "rotation": 1.0,         # 0=axis-aligned, 1=fully random per-shape rotation
    "noise_scale": 0.004,
    "noise_influence": 0.4,  # how strongly noise modulates size
    "filled": True,
    "stroke_width": 1.5,
    "alpha_min": 20,
    "alpha_max": 110,
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
    "invert_overlap": False,
}

# Shape type codes
_CIRCLE   = 0
_TRIANGLE = 1
_SQUARE   = 2


def _poly_pts(cx: float, cy: float, r: float, n_sides: int, angle: float) -> list:
    """Vertices of a regular polygon centred at (cx, cy) with circumradius r."""
    step = 2.0 * math.pi / n_sides
    return [(cx + r * math.cos(angle + k * step),
             cy + r * math.sin(angle + k * step))
            for k in range(n_sides)]


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render scattered circles, triangles and squares and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_circles       = int(cfg["n_circles"])
    n_triangles     = int(cfg["n_triangles"])
    n_squares       = int(cfg["n_squares"])
    n_total         = n_circles + n_triangles + n_squares
    ns              = float(cfg["noise_scale"]) / max(scale, 0.05)
    noise_inf       = float(cfg["noise_influence"])
    s_median        = float(cfg["size_median"]) * scale
    s_spread        = float(cfg["size_spread"])
    rotation        = float(cfg["rotation"])
    margin          = float(cfg["margin"])
    gravity         = float(cfg["gravity"])
    gravity_falloff = float(cfg["gravity_falloff"])
    filled          = bool(cfg["filled"])
    stroke_w        = max(1, round(float(cfg["stroke_width"])))
    a_min           = int(cfg["alpha_min"])
    a_max           = max(a_min + 1, int(cfg["alpha_max"]))
    invert_overlap  = bool(cfg["invert_overlap"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    if n_total == 0:
        return Image.new("RGB", (width, height), bg)

    rng = np.random.default_rng(seed)

    # ── 1. Positions ──────────────────────────────────────────────────────────
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_total)
    ys = rng.uniform(y_min, y_max, n_total)

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

    # ── 2. Shape types ────────────────────────────────────────────────────────
    types = np.array(
        [_CIRCLE] * n_circles + [_TRIANGLE] * n_triangles + [_SQUARE] * n_squares,
        dtype=np.int8,
    )
    # Shuffle so the painter's sort interleaves types naturally
    idx = rng.permutation(n_total)
    xs, ys, types = xs[idx], ys[idx], types[idx]

    # ── 3. Sizes (log-normal, noise-modulated) ─────────────────────────────────
    max_s = max(width, height) * 0.12
    log_s = rng.normal(math.log(max(s_median, 1.0)), s_spread, n_total)
    base_sizes = np.clip(np.exp(log_s), 1.0 * scale, max_s)

    if noise_inf > 0.0:
        noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)
        xi = np.clip(xs.astype(np.intp), 0, width  - 1)
        yi = np.clip(ys.astype(np.intp), 0, height - 1)
        nv = noise_field[yi, xi]
        modulation = np.clip(1.0 + nv * noise_inf, 0.15, 2.0)
        sizes = np.clip(base_sizes * modulation, 1.0 * scale, max_s * 1.5)
    else:
        sizes = base_sizes

    # ── 4. Alphas and rotations ───────────────────────────────────────────────
    alphas    = rng.integers(a_min, a_max, n_total)
    rot_range = rotation * math.pi          # max deviation from default orientation
    rotations = rng.uniform(-rot_range, rot_range, n_total)

    # ── 5. Draw order: largest first ──────────────────────────────────────────
    order = np.argsort(sizes)[::-1]

    img = Image.new("RGB", (width, height), bg)

    if invert_overlap:
        canvas_arr = np.array(img, dtype=np.float32)
        # Reusable mask canvas: clear and redraw for each shape
        mask_img  = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        bg_rect   = [0, 0, width, height]
        for i in order:
            x, y, r = float(xs[i]), float(ys[i]), float(sizes[i])
            t   = float(alphas[i]) / 255.0
            rot = float(rotations[i])
            mask_draw.rectangle(bg_rect, fill=0)
            kind = int(types[i])
            if kind == _CIRCLE:
                mask_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            elif kind == _TRIANGLE:
                mask_draw.polygon(_poly_pts(x, y, r, 3, -math.pi / 2 + rot), fill=255)
            else:
                mask_draw.polygon(_poly_pts(x, y, r, 4,  math.pi / 4 + rot), fill=255)
            mask = np.frombuffer(mask_img.tobytes(), dtype=np.uint8).reshape(height, width) > 127
            canvas_arr[mask] = np.clip(canvas_arr[mask] * (1.0 - 2.0 * t) + 255.0 * t, 0.0, 255.0)
        img = Image.fromarray(canvas_arr.astype(np.uint8), "RGB")
    else:
        color_lut: dict[int, tuple[int, int, int]] = {}
        for a in range(a_min, a_max):
            frac = a / 255.0
            color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * frac) for c in range(3))  # type: ignore[misc]

        draw = ImageDraw.Draw(img)
        for i in order:
            x, y, r = float(xs[i]), float(ys[i]), float(sizes[i])
            color = color_lut.get(int(alphas[i]), fg)
            rot   = float(rotations[i])
            kind  = int(types[i])
            if kind == _CIRCLE:
                bbox = [x - r, y - r, x + r, y + r]
                if filled:
                    draw.ellipse(bbox, fill=color)
                else:
                    draw.ellipse(bbox, outline=color, width=stroke_w)
            elif kind == _TRIANGLE:
                pts = _poly_pts(x, y, r, 3, -math.pi / 2 + rot)
                if filled:
                    draw.polygon(pts, fill=color)
                else:
                    draw.polygon(pts, outline=color, width=stroke_w)
            else:  # square
                pts = _poly_pts(x, y, r, 4, math.pi / 4 + rot)
                if filled:
                    draw.polygon(pts, fill=color)
                else:
                    draw.polygon(pts, outline=color, width=stroke_w)

    return img
