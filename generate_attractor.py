"""
Strange attractor: iterate a chaotic 2D map and plot the point density.

Two attractors are supported:
  Clifford:      x' = sin(a·y) + c·cos(a·x),  y' = sin(b·x) + d·cos(b·y)
  Peter de Jong: x' = sin(a·y) − cos(b·x),    y' = sin(c·x) − cos(d·y)

Speed trick: instead of one sequential chain of N steps, we run N_CHAINS
independent orbits in parallel using numpy vectorisation. After a short
warm-up each chain is on the attractor, so all chains sample the same
invariant density — the result is identical to a single long chain.

The accumulated histogram is log-scaled and gamma-corrected, then blended
from bg (low density) to fg (high density).
"""

import numpy as np
from PIL import Image

from generate import _hex_to_rgb

DEFAULTS: dict = {
    "seed": 42,               # unused — attractor is deterministic; kept for API parity
    "output_width": 1200,
    "output_height": 800,
    "attractor": "clifford",  # "clifford" | "dejong"
    "a":  1.7,
    "b":  1.8,
    "c": -1.9,
    "d": -0.4,
    "n_iter": 3_000_000,
    "gamma": 0.4,             # <1 brightens midtones, >1 darkens
    # Shared keys required by shared UI sections
    "margin": 0.0,
    "gravity": 0.0,
    "gravity_falloff": 0.0,
    "bg_hex": "#f5f5f0",
    "fg_hex": "#141419",
}

_SKIP    = 500   # warm-up steps discarded per chain
_NCHAINS = 128   # parallel chains; keeps numpy arrays at a workable size


def generate(config: dict, scale: float = 1.0) -> Image.Image:
    """
    Iterate a strange attractor and render its density field.

    scale < 1.0 reduces both canvas size and iteration count proportionally.
    """
    cfg = {**DEFAULTS, **config}

    width  = max(10, int(cfg["output_width"]  * scale))
    height = max(10, int(cfg["output_height"] * scale))
    kind   = str(cfg["attractor"]).lower().replace(" ", "").replace("_", "")
    a      = np.float32(cfg["a"])
    b      = np.float32(cfg["b"])
    c      = np.float32(cfg["c"])
    d      = np.float32(cfg["d"])
    # Scale iteration count with pixel area so previews are fast
    n_iter = max(20_000, int(cfg["n_iter"] * max(scale ** 2, 0.01)))
    gamma  = float(cfg["gamma"])

    bg = _hex_to_rgb(str(cfg["bg_hex"]))
    fg = _hex_to_rgb(str(cfg["fg_hex"]))

    # ── 1. Initialise chains with slightly varied starting points ─────────────
    n_per_chain = (n_iter + _NCHAINS - 1) // _NCHAINS
    rng = np.random.default_rng(0)
    x_arr = rng.uniform(-0.1, 0.1, _NCHAINS).astype(np.float32)
    y_arr = rng.uniform(-0.1, 0.1, _NCHAINS).astype(np.float32)

    clifford = (kind == "clifford")

    # ── 2. Warm-up: run each chain until it's on the attractor ───────────────
    for _ in range(_SKIP):
        if clifford:
            x_arr, y_arr = (np.sin(a*y_arr) + c*np.cos(a*x_arr),
                            np.sin(b*x_arr) + d*np.cos(b*y_arr))
        else:
            x_arr, y_arr = (np.sin(a*y_arr) - np.cos(b*x_arr),
                            np.sin(c*x_arr) - np.cos(d*y_arr))

    # ── 3. Collect points from all chains in parallel ─────────────────────────
    total_pts = _NCHAINS * n_per_chain
    all_x = np.empty(total_pts, dtype=np.float32)
    all_y = np.empty(total_pts, dtype=np.float32)

    for i in range(n_per_chain):
        if clifford:
            x_arr, y_arr = (np.sin(a*y_arr) + c*np.cos(a*x_arr),
                            np.sin(b*x_arr) + d*np.cos(b*y_arr))
        else:
            x_arr, y_arr = (np.sin(a*y_arr) - np.cos(b*x_arr),
                            np.sin(c*x_arr) - np.cos(d*y_arr))
        sl = slice(i * _NCHAINS, (i + 1) * _NCHAINS)
        all_x[sl] = x_arr
        all_y[sl] = y_arr

    xs, ys = all_x[:n_iter], all_y[:n_iter]

    # ── 4. Histogram ──────────────────────────────────────────────────────────
    x_lo, x_hi = float(xs.min()), float(xs.max())
    y_lo, y_hi = float(ys.min()), float(ys.max())
    pad_x = (x_hi - x_lo) * 0.02 + 1e-6
    pad_y = (y_hi - y_lo) * 0.02 + 1e-6

    grid, _, _ = np.histogram2d(
        ys, xs,
        bins=[height, width],
        range=[[y_lo - pad_y, y_hi + pad_y], [x_lo - pad_x, x_hi + pad_x]],
    )

    # ── 5. Tone map ───────────────────────────────────────────────────────────
    v = np.log1p(grid.astype(np.float32))
    v_max = v.max()
    if v_max > 0.0:
        v /= v_max
    v = np.power(v, gamma)

    # ── 6. Colour blend ───────────────────────────────────────────────────────
    img_arr = np.empty((height, width, 3), dtype=np.uint8)
    for ch in range(3):
        channel = bg[ch] + (fg[ch] - bg[ch]) * v
        img_arr[:, :, ch] = np.clip(channel, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr, "RGB")
