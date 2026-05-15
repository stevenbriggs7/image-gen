"""
Core image generation: scattered lines guided by a Perlin-style flow field.

Mathematical principles:
  - Fractal value noise builds a smooth, multi-scale angle field (coherent
    randomness — locally structured, globally varied).
  - Line lengths follow a Gaussian distribution (natural spread around a mean).
  - Starting positions are uniform-random; the flow field gives them direction.
  - Alpha is sampled uniformly and composited onto the background analytically.
  - Flow curves: strokes re-sample the field at each step, bending organically
    rather than pointing once in a fixed direction.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    "n_lines": 1500,
    "noise_scale": 0.004,    # spatial frequency of the angle field
    "angle_range": 1.0,      # fraction of 2π covered by the field
    "length_median": 60.0,   # median line length in px (log-normal)
    "length_spread": 0.6,    # log-sigma: 0.1 = uniform, 1.5 = extreme variation
    "flow_steps": 1,         # 1 = straight line; >1 = curved stroke following the field
    "margin": 0.0,           # fraction of each edge to exclude (0-0.45)
    "gravity": 0.0,          # pull toward centre: 0 = uniform, 1 = fully centred
    "gravity_falloff": 0.0,  # 0 = uniform effect, 1 = only nearby marks affected
    "stroke_width_min": 1.0,
    "stroke_width_max": 2.5,
    "alpha_min": 50,         # 0-255
    "alpha_max": 180,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
    "invert_overlap": False,
    "stroke_taper": 0.0,
    "stroke_width_var": 0.0,
    "stroke_break_density": 0.0,
    "stroke_roughness": 0.0,
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ── Stroke character utilities ────────────────────────────────────────────────────────────────

def _roughen_path(pts: np.ndarray, amplitude: float, rng) -> np.ndarray:
    """Perpendicular Gaussian noise on each interior point. pts shape: (N, 2)."""
    if amplitude <= 0.0 or len(pts) < 2:
        return pts
    tangents = np.diff(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    tangents /= norms
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    pts = pts.copy()
    pts[1:] += perps * rng.normal(0.0, amplitude, len(perps))[:, np.newaxis]
    return pts


def _taper_width(base_w: float, t: float, taper: float) -> int:
    """Width at fractional position t ∈ [0,1] along a stroke."""
    if taper <= 0.0:
        return max(1, round(base_w))
    factor = math.sin(math.pi * t) ** taper
    return max(1, round(base_w * max(0.12, factor)))


# ── Noise field ──────────────────────────────────────────────────────────────────────────────

def _smooth_layer(width: int, height: int, grid_scale: float, seed: int) -> np.ndarray:
    """One octave of value noise: random coarse grid + smooth bilinear upsample."""
    gw = max(2, int(math.ceil(width * grid_scale)) + 2)
    gh = max(2, int(math.ceil(height * grid_scale)) + 2)
    rng = np.random.default_rng(seed)
    grid = rng.uniform(-1.0, 1.0, (gh, gw))

    gy = np.linspace(0.0, gh - 1.0, height)
    gx = np.linspace(0.0, gw - 1.0, width)

    iy = np.clip(np.floor(gy).astype(np.intp), 0, gh - 2)
    ix = np.clip(np.floor(gx).astype(np.intp), 0, gw - 2)
    fy = gy - iy
    fx = gx - ix

    # Smoothstep (Ken Perlin's fade)
    uy = fy * fy * (3.0 - 2.0 * fy)
    ux = fx * fx * (3.0 - 2.0 * fx)

    v00 = grid[np.ix_(iy,     ix    )]
    v10 = grid[np.ix_(iy,     ix + 1)]
    v01 = grid[np.ix_(iy + 1, ix    )]
    v11 = grid[np.ix_(iy + 1, ix + 1)]

    ux2d = ux[np.newaxis, :]
    uy2d = uy[:, np.newaxis]

    return (v00 * (1 - ux2d) * (1 - uy2d)
            + v10 * ux2d      * (1 - uy2d)
            + v01 * (1 - ux2d) * uy2d
            + v11 * ux2d       * uy2d)


def _fractal_noise_field(
    width: int, height: int, noise_scale: float, seed: int,
    octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0,
) -> np.ndarray:
    """Sum of smooth layers at increasing frequencies -> fractal noise (H, W)."""
    field = np.zeros((height, width))
    amp, total, freq = 1.0, 0.0, noise_scale
    for k in range(octaves):
        field += _smooth_layer(width, height, freq, seed + k * 7919) * amp
        total += amp
        amp *= persistence
        freq *= lacunarity
    return field / total


def _invert_segment(canvas_arr, x0, y0, x1, y1, w, t, width, height):
    """Apply alpha-weighted inversion along one line segment (bounding-box fast path)."""
    pad = int(w) + 2
    xl = max(0, int(min(x0, x1)) - pad)
    xh = min(width,  int(max(x0, x1)) + pad + 1)
    yl = max(0, int(min(y0, y1)) - pad)
    yh = min(height, int(max(y0, y1)) + pad + 1)
    if xl >= xh or yl >= yh:
        return
    sy, sx = np.mgrid[yl:yh, xl:xh]
    fsx = sx.astype(np.float32)
    fsy = sy.astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    lsq = dx * dx + dy * dy
    if lsq < 1.0:
        dist = np.sqrt((fsx - x0) ** 2 + (fsy - y0) ** 2)
    else:
        tp = np.clip(((fsx - x0) * dx + (fsy - y0) * dy) / lsq, 0.0, 1.0)
        dist = np.sqrt((fsx - x0 - tp * dx) ** 2 + (fsy - y0 - tp * dy) ** 2)
    sub_mask = dist <= w * 0.5
    sub = canvas_arr[yl:yh, xl:xh]
    sub[sub_mask] = np.clip(sub[sub_mask] * (1.0 - 2.0 * t) + 255.0 * t, 0.0, 255.0)


# ── Main generation entry point ───────────────────────────────────────────────────────────

def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Render scattered lines and return a PIL Image.

    scale < 1.0 produces a proportionally smaller image (useful for previews)
    while preserving the spatial character of the noise field.
    """
    cfg = {**DEFAULTS, **config}

    seed     = int(cfg["seed"])
    width    = max(10, int(cfg["output_width"]  * scale))
    height   = max(10, int(cfg["output_height"] * scale))
    n_lines  = int(cfg["n_lines"])
    ns            = float(cfg["noise_scale"]) / max(scale, 0.05)
    a_range       = float(cfg["angle_range"]) * 2.0 * math.pi
    l_median      = float(cfg["length_median"]) * scale
    l_spread      = float(cfg["length_spread"])
    flow_steps    = max(1, int(cfg["flow_steps"]))
    margin          = float(cfg["margin"])
    gravity         = float(cfg["gravity"])
    gravity_falloff = float(cfg["gravity_falloff"])
    sw_min          = float(cfg["stroke_width_min"])
    sw_max          = float(cfg["stroke_width_max"])
    a_min           = int(cfg["alpha_min"])
    a_max           = max(a_min + 1, int(cfg["alpha_max"]))
    invert_overlap  = bool(cfg["invert_overlap"])
    taper    = float(cfg.get("stroke_taper", 0.0))
    width_var = float(cfg.get("stroke_width_var", 0.0))
    break_p  = float(cfg.get("stroke_break_density", 0.0))
    roughness = float(cfg.get("stroke_roughness", 0.0))

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    rng = np.random.default_rng(seed)

    x_min, x_max = width  * margin, width  * (1.0 - margin)
    y_min, y_max = height * margin, height * (1.0 - margin)
    xs = rng.uniform(x_min, x_max, n_lines)
    ys = rng.uniform(y_min, y_max, n_lines)
    if gravity > 0.0:
        cx, cy = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        dx, dy = cx - xs, cy - ys
        if gravity_falloff > 0.0:
            canvas_r = 0.5 * math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            d = np.sqrt(dx ** 2 + dy ** 2)
            weight = np.exp(-gravity_falloff * 5.0 * d / (canvas_r + 1e-9))
        else:
            weight = 1.0
        effective = math.sqrt(gravity)
        xs = xs + dx * effective * weight
        ys = ys + dy * effective * weight

    log_lengths = rng.normal(math.log(max(l_median, 1.0)), l_spread, n_lines)
    lengths = np.clip(np.exp(log_lengths), 3.0 * scale, max(width, height) * 1.5)

    widths = rng.uniform(sw_min, sw_max, n_lines)
    alphas = rng.integers(a_min, a_max, n_lines)

    noise_field = _fractal_noise_field(width, height, ns, seed=seed + 99991)

    xi = np.clip(xs.astype(np.intp), 0, width  - 1)
    yi = np.clip(ys.astype(np.intp), 0, height - 1)
    nv = noise_field[yi, xi]
    angles = (nv + 1.0) * 0.5 * a_range

    img = Image.new("RGB", (width, height), bg)

    if flow_steps == 1:
        # Straight line: classic vectorised endpoint computation
        halves = lengths * 0.5
        x0s = xs - np.cos(angles) * halves
        y0s = ys - np.sin(angles) * halves
        x1s = xs + np.cos(angles) * halves
        y1s = ys + np.sin(angles) * halves

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
                base_w = float(widths[i])
                if width_var > 0.0:
                    base_w *= max(0.1, 1.0 + width_var * (rng.random() - 0.5) * 2.0)
                w = _taper_width(base_w, 0.5, taper)
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
    else:
        # Flow curve: trace each stroke through the field step by step.
        # All strokes are advanced in parallel (vectorised per step) then drawn.
        step_sizes = lengths / flow_steps
        x_curr = xs.copy()
        y_curr = ys.copy()
        cur_angles = angles.copy()

        # all_pts shape: (n_lines, flow_steps+1, 2)
        all_pts = np.empty((n_lines, flow_steps + 1, 2), dtype=np.float32)
        all_pts[:, 0, 0] = x_curr
        all_pts[:, 0, 1] = y_curr

        for s in range(flow_steps):
            x_curr = x_curr + np.cos(cur_angles) * step_sizes
            y_curr = y_curr + np.sin(cur_angles) * step_sizes
            all_pts[:, s + 1, 0] = x_curr
            all_pts[:, s + 1, 1] = y_curr
            if s < flow_steps - 1:
                xi_s = np.clip(x_curr.astype(np.intp), 0, width  - 1)
                yi_s = np.clip(y_curr.astype(np.intp), 0, height - 1)
                cur_angles = (noise_field[yi_s, xi_s] + 1.0) * 0.5 * a_range

        if invert_overlap:
            canvas_arr = np.array(img, dtype=np.float32)
            for i in range(n_lines):
                t = float(alphas[i]) / 255.0
                w = max(1.0, float(widths[i]))
                for s in range(flow_steps):
                    _invert_segment(
                        canvas_arr,
                        float(all_pts[i, s,   0]), float(all_pts[i, s,   1]),
                        float(all_pts[i, s+1, 0]), float(all_pts[i, s+1, 1]),
                        w, t, width, height,
                    )
            img = Image.fromarray(canvas_arr.astype(np.uint8), "RGB")
        else:
            color_lut = {}
            for a in range(a_min, a_max):
                t = a / 255.0
                color_lut[a] = tuple(int(bg[c] + (fg[c] - bg[c]) * t) for c in range(3))  # type: ignore[misc]
            draw = ImageDraw.Draw(img)
            for i in range(n_lines):
                if break_p > 0.0 and rng.random() < break_p:
                    continue
                base_w = float(widths[i])
                if width_var > 0.0:
                    base_w *= max(0.1, 1.0 + width_var * (rng.random() - 0.5) * 2.0)
                w = _taper_width(base_w, 0.5, taper)
                if roughness > 0.0:
                    stroke_arr = _roughen_path(all_pts[i].astype(np.float64), roughness * w, rng)
                    pts = [(float(stroke_arr[s, 0]), float(stroke_arr[s, 1])) for s in range(flow_steps + 1)]
                else:
                    pts = [(float(all_pts[i, s, 0]), float(all_pts[i, s, 1])) for s in range(flow_steps + 1)]
                draw.line(pts, fill=color_lut.get(int(alphas[i]), fg), width=w)

    return img
