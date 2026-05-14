"""
Generative circles: scattered, noise-modulated rings and discs.

Mathematical principles:
  - Same fractal value noise field as the lines generator.
  - Noise drives radius modulation: coherent regions of large/small circles
    emerge naturally from the field, echoing biological cell arrangements.
  - Radii follow a log-normal base distribution (heavy tail = rare large circles).
  - Circles are drawn largest-first so fine marks read on top -- painter's algorithm.
  - Mark colour is pre-blended against background by the alpha fraction, giving
    correct opacity without requiring slow per-circle layer compositing.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _fractal_noise_field, _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_circles": 700,
    "noise_scale": 0.004,      # spatial frequency of the radius-modulation field
    "noise_influence": 0.5,    # 0 = pure log-normal, 1 = field doubles/halves radii
    "radius_median": 14.0,     # median radius in px at full resolution
    "radius_spread": 0.65,     # log-sigma: 0.2 = uniform, 1.2 = extreme variation
    "margin": 0.0,             # fraction of each edge to exclude (0-0.45)
    "gravity": 0.0,            # pull toward centre: 0 = uniform, 1 = fully centred
    "gravity_falloff": 0.0,    # 0 = uniform effect, 1 = only nearby marks affected
    "filled": True,            # True = solid disc, False = ring outline
    "stroke_width": 1.5,       # ring thickness when filled=False
    "alpha_min": 20,
    "alpha_max": 110,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
    "invert_overlap": False,
}


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render scattered circles and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews)
    while preserving the spatial character of the noise field.
    """
    cfg = {**DEFAULTS, **config}

    seed            = int(cfg["seed"])
    width           = max(10, int(cfg["output_width"]  * scale))
    height          = max(10, int(cfg["output_height"] * scale))
    n_circles       = int(cfg["n_circles"])
    ns              = float(cfg["noise_scale"]) / max(scale, 0.05)
    noise_inf       = float(cfg["noise_influence"])
    r_median        = float(cfg["radius_median"]) * scale
    r_spread        = float(cfg["radius_spread"])
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

    rng = np.random.default_rng(seed)

    # Positions constrained by margin, then optionally pulled toward centre
    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_circles)
    ys = rng.uniform(y_min, y_max, n_circles)
    if gravity > 0.0:
        cx, cy = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        dx, dy = cx - xs, cy - ys
        if gravity_falloff > 0.0:
            canvas_r = 0.5 * math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            d = np.sqrt(dx ** 2 + dy ** 2)
            weight = np.exp(-gravity_falloff * 5.0 * d / (canvas_r + 1e-9))
        else:
            weight = 1.0
        # Square-root curve: even a small slider value produces a visible pull
        effective = math.sqrt(gravity)
        xs = xs + dx * effective * weight
        ys = ys + dy * effective * weight

    # Base radii: log-normal
    log_r = rng.normal(math.log(max(r_median, 1.0)), r_spread, n_circles)
    max_r = max(width, height) * 0.12
    base_radii = np.clip(np.exp(log_r), 1.0 * scale, max_r)

    # Modulate radii by the noise field at each circle's position.
    # Noise values approx [-1, 1]; map to a multiplier centred at 1.0.
    noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)
    xi = np.clip(xs.astype(np.intp), 0, width  - 1)
    yi = np.clip(ys.astype(np.intp), 0, height - 1)
    nv = noise_field[yi, xi]
    modulation = np.clip(1.0 + nv * noise_inf, 0.15, 2.0)
    radii = np.clip(base_radii * modulation, 1.0 * scale, max_r * 1.5)

    # Alphas
    alphas = rng.integers(a_min, a_max, n_circles)

    # Draw largest circles first so small ones appear on top
    order = np.argsort(radii)[::-1]

    img = Image.new("RGB", (width, height), bg)

    if invert_overlap:
        # XOR-style inversion: each circle flips the tone of pixels beneath it.
        # Overlapping circles cancel, creating a photographic-negative effect at intersections.
        # Pure numpy bounding-box approach keeps this fast even at high circle counts.
        canvas_arr = np.array(img, dtype=np.float32)
        for i in order:
            x, y, r = float(xs[i]), float(ys[i]), float(radii[i])
            t = float(alphas[i]) / 255.0

            xl = max(0, int(x - r) - 1)
            xh = min(width,  int(x + r) + 2)
            yl = max(0, int(y - r) - 1)
            yh = min(height, int(y + r) + 2)
            if xl >= xh or yl >= yh:
                continue

            sy, sx = np.mgrid[yl:yh, xl:xh]
            dist_sq = (sx.astype(np.float32) - x) ** 2 + (sy.astype(np.float32) - y) ** 2
            sub_mask = dist_sq <= r * r

            sub = canvas_arr[yl:yh, xl:xh]
            sub[sub_mask] = np.clip(sub[sub_mask] * (1.0 - 2.0 * t) + 255.0 * t, 0.0, 255.0)

        img = Image.fromarray(canvas_arr.astype(np.uint8), "RGB")
    else:
        # Pre-blend each alpha level against the background.
        # Pillow's ImageDraw does not composite alpha, so we encode opacity as a
        # lighter/darker shade rather than using the RGBA channel.
        color_lut: dict[int, tuple[int, int, int]] = {}
        for a in range(a_min, a_max):
            t = a / 255.0
            color_lut[a] = tuple(int(bg[j] + (fg[j] - bg[j]) * t) for j in range(3))  # type: ignore[misc]

        draw = ImageDraw.Draw(img)
        for i in order:
            x, y, r = float(xs[i]), float(ys[i]), float(radii[i])
            color = color_lut.get(int(alphas[i]), fg)
            bbox = [x - r, y - r, x + r, y + r]
            if filled:
                draw.ellipse(bbox, fill=color)
            else:
                draw.ellipse(bbox, outline=color, width=stroke_w)

    return img
