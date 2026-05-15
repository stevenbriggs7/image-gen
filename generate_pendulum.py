"""
Pendulum pour: rosette pattern from a spin-launched pendulum over a canvas.

Physical model: a paint pot on a rope is spun and released. The x and y axes
of the pendulum have a tiny natural frequency mismatch (ε) — no real pivot is
perfectly isotropic. That mismatch causes the elliptical orbit to precess
(rotate) on every loop, building up the characteristic petal/rosette patterns
seen in real pendulum paintings.

Equations (anisotropic damped oscillator):
  x(t) = A         · cos(ω₀·t)           · exp(−γ·t)
  y(t) = A · aspect · sin(ω₀·(1+ε)·t + φ) · exp(−γ·t)

Approximate petal count ≈ 1/ε.
Period is independent of mass, so the pot emptying only changes amplitude
decay — already captured by γ (damping) and flow_rate (paint opacity).
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
    "precession": 0.08,  # ε — frequency mismatch; ≈1/ε petals form in the rosette
    "aspect": 1.0,       # y/x amplitude ratio: 1.0 = circular orbit, <1 = squashed
    "phase": 0.25,       # initial phase offset (0–1 → 0–2π); 0.25 = circular start
    "amplitude": 0.88,   # initial swing as fraction of min(width, height)/2
    "damping": 0.0003,   # amplitude decay rate — tighter spiral at higher values
    "n_steps": 5000,     # simulation steps (more = more loops before centre)
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
    Simulate a precessing-ellipse pendulum and render the rosette paint trail.

    scale < 1.0 produces a proportionally smaller image (useful for previews).
    """
    cfg = {**DEFAULTS, **config}

    width      = max(10, int(cfg["output_width"]  * scale))
    height     = max(10, int(cfg["output_height"] * scale))
    precession = float(cfg["precession"])
    aspect     = float(cfg["aspect"])
    phase_rad  = float(cfg["phase"]) * 2.0 * math.pi
    amplitude  = float(cfg["amplitude"])
    damping    = float(cfg["damping"])
    n_steps    = int(cfg["n_steps"])
    flow_rate  = float(cfg["flow_rate"])
    stroke_w   = max(1, round(float(cfg["stroke_width"])))
    alpha_max  = int(cfg["alpha_max"])
    alpha_min  = int(cfg["alpha_min"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    # ── 1. Time array ─────────────────────────────────────────────────────────
    dt = 0.05
    t  = np.arange(n_steps, dtype=np.float64) * dt

    # ── 2. Frequencies ────────────────────────────────────────────────────────
    omega_0 = 1.0
    omega_y = omega_0 * (1.0 + precession)   # tiny mismatch → precession

    # ── 3. Precessing ellipse path ────────────────────────────────────────────
    cx, cy = width * 0.5, height * 0.5
    R      = min(width, height) * 0.5 * amplitude
    decay  = np.exp(-damping * t)

    xs = cx + R          * np.cos(omega_0 * t)                * decay
    ys = cy + R * aspect * np.sin(omega_y  * t + phase_rad)   * decay

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
