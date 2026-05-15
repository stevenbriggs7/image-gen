"""
Pendulum pour: a continuous paint trail tracing a damped harmonograph path.

Physical model: a paint pot suspended on a rope is spun and released over a
canvas. The rope creates two perpendicular oscillations (x and y). A tiny
frequency difference between them causes the pattern to slowly precess — the
"spinning" effect. The amplitude decays exponentially as energy is lost, and
the paint grows fainter as the pot empties.

Equations:
  x(t) = A · sin(f₁·t + φ) · exp(−d·t)
  y(t) = A · sin(f₂·t)     · exp(−d·t)

where f₁ = 1 + freq_delta, f₂ = freq_ratio, and d = damping.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from generate import _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,
    "output_width": 1200,
    "output_height": 800,
    # Pendulum physics
    "freq_ratio": 1.0,      # f₂/f₁ — 1=ellipse, 2=figure-8, 1.5=trefoil-ish
    "freq_delta": 0.004,    # tiny offset on f₁ causing slow precession
    "phase": 0.25,          # initial phase difference 0–1 → 0–2π (0=line, 0.25=circle)
    "damping": 0.0003,      # amplitude decay rate per unit time
    "n_steps": 60000,       # simulation steps (more = longer trail / more loops)
    "time_scale": 1.0,      # dt multiplier
    "amplitude": 0.88,      # initial swing as fraction of min(width, height)/2
    # Paint
    "flow_rate": 0.7,       # opacity fade rate (0=constant, 2=fast)
    "stroke_width": 2.0,
    "alpha_max": 200,
    "alpha_min": 8,
    # Shared (unused by this generator but required by the shared UI sections)
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    # Colours
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
}

# Steps per rendered polyline chunk — balances draw-call overhead vs colour fidelity
_CHUNK = 200


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Simulate a damped harmonograph and render the paint trail.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    width       = max(10, int(cfg["output_width"]  * scale))
    height      = max(10, int(cfg["output_height"] * scale))
    freq_ratio  = float(cfg["freq_ratio"])
    freq_delta  = float(cfg["freq_delta"])
    phase_rad   = float(cfg["phase"]) * 2.0 * math.pi
    damping     = float(cfg["damping"])
    n_steps     = int(cfg["n_steps"])
    time_scale  = float(cfg["time_scale"])
    amplitude   = float(cfg["amplitude"])
    flow_rate   = float(cfg["flow_rate"])
    stroke_w    = max(1, round(float(cfg["stroke_width"])))
    alpha_max   = int(cfg["alpha_max"])
    alpha_min   = int(cfg["alpha_min"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    # ── 1. Time array ─────────────────────────────────────────────────────────
    dt = 0.05 * time_scale
    t  = np.arange(n_steps, dtype=np.float64) * dt

    # ── 2. Frequencies ────────────────────────────────────────────────────────
    f1 = 1.0 + freq_delta   # x frequency — precession lives here
    f2 = freq_ratio         # y frequency

    # ── 3. Path ───────────────────────────────────────────────────────────────
    cx, cy   = width * 0.5, height * 0.5
    half_w   = cx * amplitude
    half_h   = cy * amplitude
    decay    = np.exp(-damping * t)

    xs = cx + half_w * np.sin(f1 * t + phase_rad) * decay
    ys = cy + half_h * np.sin(f2 * t)             * decay

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
