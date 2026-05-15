"""
Streamlit UI -- generative image tools.

Run:
    streamlit run app.py
"""

import colorsys
import io
import random
import streamlit as st
from PIL import Image

import generate as gen_lines
import generate_circles as gen_circles
import generate_wave as gen_wave
import generate_pendulum as gen_pendulum
import generate_shapes as gen_shapes
import generate_attractor as gen_attractor
import generate_streamlines as gen_streamlines
import generate_voronoi as gen_voronoi
import generate_cubes as gen_cubes
import generate_cubes_grid as gen_cubes_grid
import generate_spirograph as gen_spirograph


# ── Mood colour system ─────────────────────────────────────────────────────────
# Each mood defines HLS ranges for background and mark colour.
# Foreground hue is derived from background via one of three strategies:
#   "analogous"  – same hue ± small offset (harmonious, related tones)
#   "complement" – opposite hue ± offset (vibrant contrast)
#   "fixed"      – independent hue range (for moods where bg hue is arbitrary)

_MOODS: dict = {
    "ink": {
        "bg": {"h": (30, 220), "s": (0.00, 0.06), "l": (0.93, 0.99)},
        "fg": {"strategy": "analogous", "offset": (-15, 15), "s": (0.00, 0.10), "l": (0.02, 0.10)},
    },
    "parchment": {
        "bg": {"h": (35, 55), "s": (0.28, 0.55), "l": (0.84, 0.94)},
        "fg": {"strategy": "analogous", "offset": (-20, 10), "s": (0.38, 0.68), "l": (0.12, 0.26)},
    },
    "fog": {
        "bg": {"h": (170, 230), "s": (0.04, 0.14), "l": (0.78, 0.90)},
        "fg": {"strategy": "analogous", "offset": (-15, 15), "s": (0.06, 0.20), "l": (0.18, 0.36)},
    },
    "sand": {
        "bg": {"h": (22, 45), "s": (0.38, 0.65), "l": (0.72, 0.88)},
        "fg": {"strategy": "analogous", "offset": (-18, 12), "s": (0.48, 0.72), "l": (0.16, 0.30)},
    },
    "dawn": {
        "bg": {"h": (10, 52), "s": (0.48, 0.78), "l": (0.82, 0.94)},
        "fg": {"strategy": "analogous", "offset": (-25, 25), "s": (0.52, 0.80), "l": (0.13, 0.28)},
    },
    "ember": {
        "bg": {"h": (12, 30), "s": (0.22, 0.48), "l": (0.07, 0.15)},
        "fg": {"strategy": "analogous", "offset": (-12, 20), "s": (0.78, 1.00), "l": (0.52, 0.72)},
    },
    "dusk": {
        "bg": {"h": (255, 288), "s": (0.38, 0.68), "l": (0.09, 0.19)},
        "fg": {"strategy": "fixed", "h": (12, 50), "s": (0.68, 0.92), "l": (0.58, 0.78)},
    },
    "midnight": {
        "bg": {"h": (215, 265), "s": (0.42, 0.72), "l": (0.05, 0.14)},
        "fg": {"strategy": "complement", "offset": (162, 198), "s": (0.55, 0.82), "l": (0.60, 0.82)},
    },
    "storm": {
        "bg": {"h": (195, 228), "s": (0.16, 0.36), "l": (0.11, 0.22)},
        "fg": {"strategy": "analogous", "offset": (-22, 22), "s": (0.08, 0.28), "l": (0.80, 0.94)},
    },
    "arctic": {
        "bg": {"h": (175, 215), "s": (0.18, 0.42), "l": (0.87, 0.96)},
        "fg": {"strategy": "analogous", "offset": (-15, 15), "s": (0.48, 0.78), "l": (0.15, 0.32)},
    },
    "forest": {
        "bg": {"h": (112, 148), "s": (0.42, 0.68), "l": (0.09, 0.20)},
        "fg": {"strategy": "fixed", "h": (35, 68), "s": (0.68, 0.92), "l": (0.58, 0.80)},
    },
    "neon": {
        "bg": {"h": (0, 360), "s": (0.00, 0.18), "l": (0.04, 0.09)},
        "fg": {"strategy": "fixed", "h": (0, 360), "s": (0.88, 1.00), "l": (0.48, 0.64)},
    },
}

_MOOD_LABELS = {
    "ink":       "Ink",
    "parchment": "Parchment",
    "fog":       "Fog",
    "sand":      "Sand",
    "dawn":      "Dawn",
    "ember":     "Ember",
    "dusk":      "Dusk",
    "midnight":  "Midnight",
    "storm":     "Storm",
    "arctic":    "Arctic",
    "forest":    "Forest",
    "neon":      "Neon",
}


def _hls_hex(h_deg: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0 % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def mood_colors(mood: str, seed: int) -> tuple[str, str]:
    """Return (bg_hex, fg_hex) drawn from the mood's colour ranges using seed."""
    spec = _MOODS.get(mood, _MOODS["ink"])
    rng = random.Random(seed)

    def u(lo: float, hi: float) -> float:
        return lo + rng.random() * (hi - lo)

    bg   = spec["bg"]
    bg_h = u(*bg["h"]) % 360.0
    bg_s = u(*bg["s"])
    bg_l = u(*bg["l"])

    fg       = spec["fg"]
    strategy = fg["strategy"]
    if strategy == "complement":
        fg_h = (bg_h + u(*fg["offset"])) % 360.0
    elif strategy == "analogous":
        fg_h = (bg_h + u(*fg["offset"])) % 360.0
    else:  # "fixed" — independent hue range
        fg_h = u(*fg["h"]) % 360.0
    fg_s = u(*fg["s"])
    fg_l = u(*fg["l"])

    return _hls_hex(bg_h, bg_l, bg_s), _hls_hex(fg_h, fg_l, fg_s)


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Generative Art",
    page_icon="o",
    layout="wide",
)

st.markdown("""
<style>
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; }
[data-testid="stImage"] img { max-height: 42vh; width: 100%; object-fit: contain; }
</style>
""", unsafe_allow_html=True)

# Session state
if "art_seed" not in st.session_state:
    st.session_state.art_seed = 42
if "colour_seed" not in st.session_state:
    st.session_state.colour_seed = random.randint(0, 99999)


def _reroll_colours() -> None:
    st.session_state.colour_seed = random.randint(0, 99999)


# Image placeholder at top — filled after params are collected
img_placeholder     = st.empty()
caption_placeholder = st.empty()

# ── Controls panel ─────────────────────────────────────────────────────────────

with st.container(height=500, border=False):

    # Type picker
    gen_type = st.radio(
        "Type", ["Lines", "Circles", "Wave", "Pendulum", "Shapes", "Attractor", "Streamlines", "Voronoi", "Cubes", "Grid", "Spirograph"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # Seed (shared)
    defaults = (gen_lines.DEFAULTS          if gen_type == "Lines"
                else gen_circles.DEFAULTS       if gen_type == "Circles"
                else gen_wave.DEFAULTS          if gen_type == "Wave"
                else gen_pendulum.DEFAULTS      if gen_type == "Pendulum"
                else gen_shapes.DEFAULTS        if gen_type == "Shapes"
                else gen_attractor.DEFAULTS     if gen_type == "Attractor"
                else gen_streamlines.DEFAULTS   if gen_type == "Streamlines"
                else gen_voronoi.DEFAULTS       if gen_type == "Voronoi"
                else gen_cubes.DEFAULTS         if gen_type == "Cubes"
                else gen_cubes_grid.DEFAULTS    if gen_type == "Grid"
                else gen_spirograph.DEFAULTS)
    # Seed — shown only for generators with random variation
    _SEED_TYPES = {"Lines", "Circles", "Wave", "Shapes", "Streamlines", "Voronoi", "Cubes", "Grid"}
    seed = st.session_state.art_seed
    if gen_type in _SEED_TYPES:
        col_btn, col_val = st.columns([2, 3])
        with col_btn:
            if st.button("Randomize", use_container_width=True):
                st.session_state.art_seed = random.randint(0, 99999)
                seed = st.session_state.art_seed
        with col_val:
            st.write("")
            st.caption(f"seed {seed}")

    # ── Lines controls ────────────────────────────────────────────────────────
    if gen_type == "Lines":
        D = gen_lines.DEFAULTS

        st.subheader("Lines")
        n_lines       = st.slider("Count", 50, 5000, D["n_lines"], step=50, key="l_count")
        length_median = st.slider("Length median (px)", 5, 400, int(D["length_median"]), key="l_len_med")
        length_spread = st.slider(
            "Length variation", 0.1, 2.0, D["length_spread"], step=0.05, format="%.2f",
            help="Log-normal spread. 0.1 = uniform, 1.5 = extreme.",
            key="l_len_spread",
        )

        st.subheader("Flow Field")
        noise_scale = st.slider(
            "Noise scale", 0.0005, 0.025, D["noise_scale"],
            step=0.0005, format="%.4f",
            help="Low = sweeping coherent directions. High = chaotic.",
            key="l_noise_scale",
        )
        angle_range = st.slider(
            "Angle range (x 2pi)", 0.05, 1.0, D["angle_range"],
            step=0.05, format="%.2f",
            help="1.0 = any direction; 0.25 = constrained to a quadrant.",
            key="l_angle_range",
        )

        st.subheader("Stroke")
        sw_min, sw_max = st.slider(
            "Width range (px)", 0.5, 8.0,
            (D["stroke_width_min"], D["stroke_width_max"]),
            step=0.5, key="l_sw",
        )
        alpha_min, alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="l_alpha",
        )
        flow_steps = st.slider(
            "Curve steps", 1, 20, D["flow_steps"], step=1,
            help="1 = straight lines. Higher values trace each stroke through the flow field, creating organic curves.",
            key="l_flow",
        )
        invert_overlap = st.toggle(
            "Invert overlap",
            value=D["invert_overlap"],
            help="Each line inverts the tone beneath it. Overlapping lines cancel, creating lattice and moiré effects.",
            key="l_invert",
        )

    # ── Circles controls ──────────────────────────────────────────────────────
    elif gen_type == "Circles":
        D = gen_circles.DEFAULTS

        st.subheader("Circles")
        n_circles     = st.slider("Count", 50, 3000, D["n_circles"], step=50, key="c_count")
        radius_median = st.slider("Radius median (px)", 2, 200, int(D["radius_median"]), key="c_rad_med")
        radius_spread = st.slider(
            "Radius variation", 0.1, 2.0, D["radius_spread"], step=0.05, format="%.2f",
            help="Log-normal spread. 0.2 = uniform, 1.2 = extreme range.",
            key="c_rad_spread",
        )

        st.subheader("Noise Field")
        noise_scale = st.slider(
            "Noise scale", 0.0005, 0.025, D["noise_scale"],
            step=0.0005, format="%.4f",
            help="Controls the spatial scale of radius clustering.",
            key="c_noise_scale",
        )
        noise_influence = st.slider(
            "Noise influence", 0.0, 1.5, D["noise_influence"],
            step=0.05, format="%.2f",
            help="How strongly the field modulates radius. 0 = pure random.",
            key="c_noise_inf",
        )

        st.subheader("Style")
        filled = st.toggle("Filled discs", value=D["filled"],
                           help="Off = ring outlines.", key="c_filled")
        if not filled:
            stroke_w = st.slider("Ring thickness (px)", 0.5, 6.0, D["stroke_width"],
                                 step=0.5, key="c_stroke_w")
        else:
            stroke_w = D["stroke_width"]

        alpha_min, alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="c_alpha",
        )
        invert_overlap = st.toggle(
            "Invert overlap",
            value=D["invert_overlap"],
            help="Each circle inverts the tone beneath it. Overlapping circles cancel, showing the negative at intersections.",
            key="c_invert",
        )
        flow_steps = 1  # not applicable to circles

    # ── Wave controls ─────────────────────────────────────────────────────────
    elif gen_type == "Wave":
        D = gen_wave.DEFAULTS

        st.subheader("Wave Lines")
        n_lines       = st.slider("Count", 50, 3000, D["n_lines"], step=50, key="w_count")
        length_median = st.slider("Length median (px)", 5, 300, int(D["length_median"]), key="w_len_med")
        length_spread = st.slider(
            "Length variation", 0.1, 2.0, D["length_spread"], step=0.05, format="%.2f",
            key="w_len_spread",
        )

        st.subheader("Wave Shape")
        wave_amplitude = st.slider(
            "Amplitude", 0.0, 0.5, D["wave_amplitude"], step=0.01, format="%.2f",
            help="Height of the wave as a fraction of canvas height.",
            key="w_amp",
        )
        wave_frequency = st.slider(
            "Frequency", 0.25, 8.0, D["wave_frequency"], step=0.25, format="%.2f",
            help="Number of complete cycles across the canvas width.",
            key="w_freq",
        )
        wave_phase = st.slider(
            "Phase", 0.0, 1.0, D["wave_phase"], step=0.05, format="%.2f",
            help="Shifts the wave left or right (0–1 wraps once around).",
            key="w_phase",
        )
        wave_angle = st.slider(
            "Wave angle (°)", 0, 180, int(D["wave_angle"]), step=5,
            help="Direction the wave travels. 0° = horizontal, 90° = vertical.",
            key="w_angle",
        )

        st.subheader("Wave Influence")
        line_scatter = st.slider(
            "Scatter", 0.0, 0.4, D["line_scatter"], step=0.01, format="%.2f",
            help="Spread line centres off the wave. 0 = all centres exactly on the wave.",
            key="w_scatter",
        )
        angle_spread = st.slider(
            "Angle variation", 0.0, 1.5, D["angle_spread"], step=0.05, format="%.2f",
            help="Angular randomness around the perpendicular (radians). 0 = all lines parallel.",
            key="w_angle_spread",
        )

        st.subheader("Stroke")
        sw_min, sw_max = st.slider(
            "Width range (px)", 0.5, 8.0,
            (D["stroke_width_min"], D["stroke_width_max"]),
            step=0.5, key="w_sw",
        )
        alpha_min, alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="w_alpha",
        )
        invert_overlap = st.toggle(
            "Invert overlap", value=D["invert_overlap"],
            help="Each line inverts the tone beneath it.",
            key="w_invert",
        )
        flow_steps = 1  # not applicable

    elif gen_type == "Pendulum":
        D = gen_pendulum.DEFAULTS

        st.subheader("Pendulum")
        precession = st.slider(
            "Precession", 0.01, 0.5, D["precession"], step=0.01, format="%.2f",
            help="How fast the orbit rotates. Lower = more petals (≈1/value). 0.1 → ~10 petals, 0.33 → ~3.",
            key="p_prec",
        )
        aspect = st.slider(
            "Shape", 0.1, 1.0, D["aspect"], step=0.05, format="%.2f",
            help="1.0 = circular orbit, lower = squashed — makes petals more elongated.",
            key="p_aspect",
        )
        phase = st.slider(
            "Phase", 0.0, 1.0, D["phase"], step=0.05, format="%.2f",
            help="Starting offset between x and y. 0.25 = circular start, 0 = straight-line start.",
            key="p_phase",
        )
        amplitude = st.slider(
            "Amplitude", 0.3, 1.0, D["amplitude"], step=0.05, format="%.2f",
            help="Initial swing size as a fraction of the canvas.",
            key="p_amp",
        )

        st.subheader("Paint")
        damping = st.slider(
            "Damping", 0.0001, 0.002, D["damping"], step=0.0001, format="%.4f",
            help="How quickly the swing decays — tighter spiral at higher values.",
            key="p_damp",
        )
        n_steps = st.slider(
            "Steps", 1, 10000, min(D["n_steps"], 10000), step=100,
            help="Simulation length — more steps draw more loops of the pattern.",
            key="p_steps",
        )
        flow_rate = st.slider(
            "Flow rate", 0.0, 2.0, D["flow_rate"], step=0.1, format="%.1f",
            help="How quickly opacity fades as the pot empties. 0=constant ink.",
            key="p_flow",
        )

        st.subheader("Stroke")
        stroke_width = st.slider(
            "Width (px)", 0.5, 8.0, D["stroke_width"], step=0.5,
            key="p_sw",
        )
        p_alpha_max = st.slider(
            "Opacity start (0-255)", 0, 255, D["alpha_max"],
            help="Opacity at the beginning of the trace.",
            key="p_alpha_max",
        )
        p_alpha_min = st.slider(
            "Opacity end (0-255)", 0, 255, D["alpha_min"],
            help="Opacity at the end of the trace (pot nearly empty).",
            key="p_alpha_min",
        )

    elif gen_type == "Shapes":
        D = gen_shapes.DEFAULTS

        st.subheader("Shape mix")
        n_circles   = st.slider("Circles",   0, 1000, D["n_circles"],   step=10, key="sh_circles")
        n_triangles = st.slider("Triangles", 0, 1000, D["n_triangles"], step=10, key="sh_triangles")
        n_squares   = st.slider("Squares",   0, 1000, D["n_squares"],   step=10, key="sh_squares")

        st.subheader("Size")
        size_median = st.slider(
            "Size median (px)", 2, 200, int(D["size_median"]), key="sh_size_med",
        )
        size_spread = st.slider(
            "Size variation", 0.1, 2.0, D["size_spread"], step=0.05, format="%.2f",
            key="sh_size_spread",
        )
        rotation = st.slider(
            "Rotation", 0.0, 1.0, D["rotation"], step=0.05, format="%.2f",
            help="0 = all axis-aligned, 1 = fully random per-shape rotation.",
            key="sh_rotation",
        )

        st.subheader("Noise")
        noise_scale = st.slider(
            "Noise scale", 0.001, 0.02, D["noise_scale"], step=0.001, format="%.3f",
            help="Spatial frequency of the size-modulation field.",
            key="sh_noise_scale",
        )
        noise_influence = st.slider(
            "Noise influence", 0.0, 1.0, D["noise_influence"], step=0.05, format="%.2f",
            help="How strongly noise modulates size. 0 = pure log-normal.",
            key="sh_noise_inf",
        )

        st.subheader("Style")
        filled = st.toggle("Filled", value=D["filled"], key="sh_filled")
        stroke_w = st.slider(
            "Stroke width (px)", 0.5, 8.0, D["stroke_width"], step=0.5, key="sh_stroke",
        )
        alpha_min, alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="sh_alpha",
        )
        invert_overlap = st.toggle(
            "Invert overlap", value=D["invert_overlap"],
            help="Each shape inverts the tone beneath it.",
            key="sh_invert",
        )

    elif gen_type == "Attractor":
        D = gen_attractor.DEFAULTS

        st.subheader("Attractor")
        attractor_type = st.selectbox(
            "Type", ["Clifford", "De Jong"],
            index=0 if D["attractor"] == "clifford" else 1,
            key="at_type",
        )
        n_iter = st.slider(
            "Iterations", 500_000, 8_000_000, D["n_iter"], step=500_000,
            help="More iterations = denser, smoother pattern. Scales down automatically for previews.",
            key="at_iter",
        )

        st.subheader("Parameters")
        a = st.slider("a", -3.0, 3.0, D["a"], step=0.05, format="%.2f", key="at_a")
        b = st.slider("b", -3.0, 3.0, D["b"], step=0.05, format="%.2f", key="at_b")
        c = st.slider("c", -3.0, 3.0, D["c"], step=0.05, format="%.2f", key="at_c")
        d = st.slider("d", -3.0, 3.0, D["d"], step=0.05, format="%.2f", key="at_d")

        st.subheader("Tone")
        gamma = st.slider(
            "Gamma", 0.1, 2.0, D["gamma"], step=0.05, format="%.2f",
            help="< 1 brightens midtones (dusty), > 1 darkens them (high-contrast).",
            key="at_gamma",
        )

    elif gen_type == "Streamlines":
        D = gen_streamlines.DEFAULTS

        st.subheader("Flow")
        n_lines = st.slider("Streams", 10, 500, D["n_lines"], step=10, key="sl_count")
        n_steps = st.slider(
            "Steps", 50, 2000, D["n_steps"], step=50,
            help="How far each stream travels. More steps = longer, more winding paths.",
            key="sl_steps",
        )
        step_size = st.slider(
            "Step size (px)", 1.0, 20.0, D["step_size"], step=0.5, format="%.1f",
            help="Distance moved each step. Larger = faster but coarser curves.",
            key="sl_step_size",
        )

        st.subheader("Field")
        noise_scale = st.slider(
            "Noise scale", 0.0005, 0.02, D["noise_scale"], step=0.0005, format="%.4f",
            help="Low = sweeping river-like paths. High = chaotic turbulence.",
            key="sl_noise_scale",
        )
        angle_range = st.slider(
            "Angle range (x 2pi)", 0.05, 1.0, D["angle_range"], step=0.05, format="%.2f",
            help="1.0 = any direction; 0.25 = constrained to a quadrant.",
            key="sl_angle_range",
        )

        st.subheader("Stroke")
        stroke_width = st.slider("Width (px)", 0.5, 6.0, D["stroke_width"], step=0.5, key="sl_sw")
        alpha_max = st.slider(
            "Opacity start (0-255)", 0, 255, D["alpha_max"],
            help="Opacity at the start of each stream (freshest ink).",
            key="sl_alpha_max",
        )
        alpha_min = st.slider(
            "Opacity end (0-255)", 0, 255, D["alpha_min"],
            help="Opacity at the end of each stream (ink thinning).",
            key="sl_alpha_min",
        )

    elif gen_type == "Voronoi":
        D = gen_voronoi.DEFAULTS

        st.subheader("Cells")
        n_cells = st.slider("Cell count", 10, 800, D["n_cells"], step=10, key="vo_count")

        st.subheader("Noise")
        noise_scale = st.slider(
            "Noise scale", 0.0005, 0.02, D["noise_scale"], step=0.0005, format="%.4f",
            help="Spatial scale of the density field. Low = large blobs, high = fine grain.",
            key="vo_noise_scale",
        )
        noise_influence = st.slider(
            "Noise influence", 0.0, 1.5, D["noise_influence"], step=0.05, format="%.2f",
            help="How strongly the field modulates cell shading. 0 = uniform.",
            key="vo_noise_inf",
        )

        st.subheader("Style")
        filled = st.toggle("Filled cells", value=D["filled"],
                           help="Off = outlines only (cracked-earth look).", key="vo_filled")
        stroke_w = st.slider("Outline width (px)", 0.5, 6.0, D["stroke_width"], step=0.5, key="vo_stroke")
        alpha_min, alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="vo_alpha",
        )

    elif gen_type == "Cubes":
        D = gen_cubes.DEFAULTS

        st.subheader("Cubes")
        n_cubes      = st.slider("Count", 20, 1000, D["n_cubes"], step=20, key="cu_count")
        size_median  = st.slider("Size median (px)", 5, 200, int(D["size_median"]), key="cu_size_med")
        size_spread  = st.slider(
            "Size variation", 0.1, 1.5, D["size_spread"], step=0.05, format="%.2f",
            key="cu_size_spread",
        )

        st.subheader("Noise")
        noise_scale = st.slider(
            "Noise scale", 0.0005, 0.02, D["noise_scale"], step=0.0005, format="%.4f",
            help="Spatial scale of size clustering. Low = large blobs, high = fine grain.",
            key="cu_noise_scale",
        )
        noise_influence = st.slider(
            "Noise influence", 0.0, 1.5, D["noise_influence"], step=0.05, format="%.2f",
            help="How strongly noise modulates cube size. 0 = pure random.",
            key="cu_noise_inf",
        )

        st.subheader("Style")
        filled = st.toggle("Filled", value=D["filled"], key="cu_filled")
        stroke_w = st.slider("Outline width (px)", 0.5, 6.0, D["stroke_width"], step=0.5, key="cu_stroke")
        alpha_min, alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="cu_alpha",
        )
        shade_contrast = st.slider(
            "Shade contrast", 0.0, 1.0, D["shade_contrast"], step=0.05, format="%.2f",
            help="How different the three face shades are. 0 = flat, 1 = strong light/shadow.",
            key="cu_shade",
        )

        st.subheader("Warp")
        cu_warp_label = st.selectbox(
            "Warp type", ["None", "Wave", "Sphere"],
            index=0,
            help="Displaces each vertex by an invisible field, bending the cube faces.",
            key="cu_warp_type",
        )
        warp_type = cu_warp_label.lower()
        if warp_type != "none":
            warp_amplitude = st.slider(
                "Warp amplitude", 0.0, 1.0, D["warp_amplitude"], step=0.05, format="%.2f",
                help="Displacement strength as a fraction of cube size.",
                key="cu_warp_amp",
            )
            warp_scale = st.slider(
                "Warp scale", 0.001, 0.02, D["warp_scale"], step=0.001, format="%.3f",
                help="Wave: spatial frequency. Sphere: falloff rate from canvas centre.",
                key="cu_warp_scale",
            )
        else:
            warp_amplitude = 0.0
            warp_scale = D["warp_scale"]

    elif gen_type == "Grid":
        D = gen_cubes_grid.DEFAULTS

        st.subheader("Grid")
        cube_size = st.slider(
            "Cube size (px)", 10, 120, int(D["cube_size"]), step=5,
            help="Height of one cube in pixels. Smaller = more cubes, denser grid.",
            key="gr_cube_size",
        )
        max_height = st.slider(
            "Max height", 1, 12, int(D["max_height"]), step=1,
            help="Maximum number of cubes stacked per column.",
            key="gr_max_h",
        )
        if max_height > 1:
            height_style = st.selectbox(
                "Height style", ["Flat", "Terrain", "Towers", "Wave"],
                index=["flat", "terrain", "towers", "wave"].index(D["height_style"]),
                help="How column heights are assigned across the grid.",
                key="gr_h_style",
            ).lower()
            if height_style == "terrain":
                height_noise_scale = st.slider(
                    "Terrain scale", 0.001, 0.02, D["height_noise_scale"],
                    step=0.001, format="%.3f",
                    help="Spatial scale of the fractal terrain. Low = broad hills, high = rugged.",
                    key="gr_h_ns",
                )
            elif height_style == "wave":
                height_noise_scale = st.slider(
                    "Wave frequency", 0.001, 0.02, D["height_noise_scale"],
                    step=0.001, format="%.3f",
                    help="Spatial frequency of the sinusoidal ripple.",
                    key="gr_h_ns",
                )
            else:
                height_noise_scale = D["height_noise_scale"]
        else:
            height_style = "flat"
            height_noise_scale = D["height_noise_scale"]

        st.subheader("Noise")
        gr_noise_scale = st.slider(
            "Tone noise scale", 0.001, 0.02, D["noise_scale"],
            step=0.001, format="%.3f",
            help="Spatial scale of the per-cube tone variation field.",
            key="gr_noise_scale",
        )
        gr_noise_influence = st.slider(
            "Tone noise influence", 0.0, 1.0, D["noise_influence"],
            step=0.05, format="%.2f",
            help="How strongly the field modulates cube brightness. 0 = uniform.",
            key="gr_noise_inf",
        )

        st.subheader("Style")
        gr_filled = st.toggle("Filled", value=D["filled"], key="gr_filled")
        gr_stroke_w = st.slider(
            "Outline width (px)", 0.5, 6.0, D["stroke_width"], step=0.5, key="gr_stroke",
        )
        gr_alpha_min, gr_alpha_max = st.slider(
            "Opacity range (0-255)", 0, 255,
            (D["alpha_min"], D["alpha_max"]),
            key="gr_alpha",
        )
        gr_shade_contrast = st.slider(
            "Shade contrast", 0.0, 1.0, D["shade_contrast"], step=0.05, format="%.2f",
            help="How different the three face shades are. 0 = flat, 1 = strong light/shadow.",
            key="gr_shade",
        )

    elif gen_type == "Spirograph":
        D = gen_spirograph.DEFAULTS

        st.subheader("Gears")
        spiro_mode = st.selectbox(
            "Type", ["Hypo (inner rolling)", "Epi (outer rolling)"],
            index=0 if D["mode"] == "hypo" else 1,
            key="sp_mode",
        )
        R_val = st.slider("Outer radius R", 2, 20, int(D["R"]), step=1,
                          help="Fixed gear radius. Number of petals ≈ R / gcd(R, r).",
                          key="sp_R")
        r_val = st.slider("Inner radius r", 1, 15, int(D["r"]), step=1,
                          help="Rolling gear radius. Curve closes after r / gcd(R, r) loops.",
                          key="sp_r")
        d_val = st.slider(
            "Pen distance d", 0.5, 20.0, float(D["d"]), step=0.5, format="%.1f",
            help="Distance of pen from rolling gear center. d = r gives a classic hypocycloid.",
            key="sp_d",
        )

        st.subheader("Trace")
        n_repeats = st.slider(
            "Repeats", 1, 5, D["n_repeats"], step=1,
            help="Retrace the closed pattern this many times for a layered, denser look.",
            key="sp_repeats",
        )

        st.subheader("Stroke")
        stroke_width = st.slider("Width (px)", 0.5, 6.0, D["stroke_width"], step=0.5, key="sp_sw")
        sp_alpha_max = st.slider("Opacity start (0-255)", 0, 255, D["alpha_max"],
                                 help="Opacity at the start of the trace.", key="sp_alpha_max")
        sp_alpha_min = st.slider("Opacity end (0-255)", 0, 255, D["alpha_min"],
                                 help="Opacity at the end of the trace.", key="sp_alpha_min")

    # ── Stroke Character (shared, stroke-based generators only) ──────────────
    _STROKE_TYPES = {"Lines", "Wave", "Streamlines", "Pendulum", "Spirograph"}
    if gen_type in _STROKE_TYPES:
        st.subheader("Stroke Character")
        stroke_taper = st.slider(
            "End taper", 0.0, 1.0, float(defaults.get("stroke_taper", 0.0)),
            step=0.05, key="sc_taper",
            help="Narrows stroke width at both ends of each stroke.",
        )
        stroke_width_var = st.slider(
            "Width variation", 0.0, 1.0, float(defaults.get("stroke_width_var", 0.0)),
            step=0.05, key="sc_wvar",
            help="Random per-stroke width jitter for a hand-drawn rhythm.",
        )
        stroke_break_density = st.slider(
            "Break density", 0.0, 0.5, float(defaults.get("stroke_break_density", 0.0)),
            step=0.02, key="sc_break",
            help="Fraction of strokes/segments randomly omitted (dry-brush gaps).",
        )
        stroke_roughness = st.slider(
            "Roughness", 0.0, 2.0, float(defaults.get("stroke_roughness", 0.0)),
            step=0.1, key="sc_rough",
            help="Perpendicular wobble amplitude (× stroke width) for hand-tremor feel.",
        )
    else:
        stroke_taper = stroke_width_var = stroke_break_density = stroke_roughness = 0.0

    # ── Composition (shared) ──────────────────────────────────────────────────
    st.subheader("Composition")
    margin = st.slider(
        "Edge margin", 0.0, 0.45, defaults["margin"],
        step=0.01, format="%.2f",
        help="Fraction of the image to leave empty at each edge.",
        key="shared_margin",
    )
    _GRAVITY_TYPES = {"Lines", "Circles", "Wave", "Shapes", "Streamlines", "Voronoi", "Cubes"}
    if gen_type in _GRAVITY_TYPES:
        gravity = st.slider(
            "Gravity strength", 0.0, 0.95, defaults["gravity"],
            step=0.05, format="%.2f",
            help="How strongly marks are pulled toward the centre.",
            key="shared_gravity",
        )
        gravity_falloff = st.slider(
            "Gravity falloff", 0.0, 1.0, defaults["gravity_falloff"],
            step=0.05, format="%.2f",
            help="How quickly gravity weakens with distance. 0 = uniform pull everywhere, 1 = only nearby marks affected.",
            key="shared_gravity_falloff",
        )
    else:
        gravity = gravity_falloff = 0.0

    # ── Mood / Color & Output (shared) ────────────────────────────────────────
    st.subheader("Mood & Output")
    col_mood, col_reroll = st.columns([3, 1])
    with col_mood:
        mood = st.selectbox(
            "Mood",
            list(_MOOD_LABELS.keys()),
            format_func=_MOOD_LABELS.get,
            key="shared_mood",
            on_change=_reroll_colours,
        )
    with col_reroll:
        st.write("")
        if st.button("Re-roll", key="shared_reroll", use_container_width=True,
                     help="Pick new colours within this mood"):
            _reroll_colours()

    bg_hex, fg_hex = mood_colors(mood, st.session_state.colour_seed)

    out_w = st.select_slider(
        "Width (px)", [400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3000, 4000],
        value=2000, key="shared_out_w",
    )
    out_h = st.select_slider(
        "Height (px)", [300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3000, 4000],
        value=2000, key="shared_out_h",
    )

    st.divider()
    col_btn2, col_tip = st.columns([1, 3])
    with col_btn2:
        render_full = st.button("Render full resolution", type="primary")
    with col_tip:
        st.caption("Preview updates automatically. Render for full-res export.")


# ── Assemble config ────────────────────────────────────────────────────────────

if gen_type == "Lines":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_lines": n_lines,
        "noise_scale": noise_scale, "angle_range": angle_range,
        "length_median": float(length_median), "length_spread": float(length_spread),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "stroke_width_min": sw_min, "stroke_width_max": sw_max,
        "alpha_min": alpha_min, "alpha_max": alpha_max,
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "flow_steps": flow_steps,
        "invert_overlap": invert_overlap,
        "stroke_taper": float(stroke_taper), "stroke_width_var": float(stroke_width_var),
        "stroke_break_density": float(stroke_break_density), "stroke_roughness": float(stroke_roughness),
    }
    label = "lines"
elif gen_type == "Wave":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_lines": n_lines,
        "wave_amplitude": float(wave_amplitude),
        "wave_frequency": float(wave_frequency),
        "wave_phase": float(wave_phase),
        "wave_angle": float(wave_angle),
        "line_scatter": float(line_scatter),
        "angle_spread": float(angle_spread),
        "length_median": float(length_median), "length_spread": float(length_spread),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "stroke_width_min": sw_min, "stroke_width_max": sw_max,
        "alpha_min": alpha_min, "alpha_max": alpha_max,
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "invert_overlap": invert_overlap,
        "stroke_taper": float(stroke_taper), "stroke_width_var": float(stroke_width_var),
        "stroke_break_density": float(stroke_break_density), "stroke_roughness": float(stroke_roughness),
    }
    label = "wave"
elif gen_type == "Pendulum":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "precession": float(precession),
        "aspect": float(aspect),
        "phase": float(phase),
        "damping": float(damping),
        "n_steps": int(n_steps),
        "amplitude": float(amplitude),
        "flow_rate": float(flow_rate),
        "stroke_width": float(stroke_width),
        "alpha_max": int(p_alpha_max),
        "alpha_min": int(p_alpha_min),
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "stroke_taper": float(stroke_taper), "stroke_width_var": float(stroke_width_var),
        "stroke_break_density": float(stroke_break_density), "stroke_roughness": float(stroke_roughness),
    }
    label = "pendulum"
elif gen_type == "Shapes":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_circles": int(n_circles), "n_triangles": int(n_triangles), "n_squares": int(n_squares),
        "size_median": float(size_median), "size_spread": float(size_spread),
        "rotation": float(rotation),
        "noise_scale": float(noise_scale), "noise_influence": float(noise_influence),
        "filled": filled, "stroke_width": float(stroke_w),
        "alpha_min": int(alpha_min), "alpha_max": int(alpha_max),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "invert_overlap": invert_overlap,
    }
    label = "shapes"
elif gen_type == "Attractor":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "attractor": attractor_type.lower().replace(" ", ""),
        "a": float(a), "b": float(b), "c": float(c), "d": float(d),
        "n_iter": int(n_iter),
        "gamma": float(gamma),
        "margin": 0.0, "gravity": 0.0, "gravity_falloff": 0.0,
        "bg_hex": bg_hex, "fg_hex": fg_hex,
    }
    label = "attractor"
elif gen_type == "Streamlines":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_lines": int(n_lines),
        "n_steps": int(n_steps),
        "step_size": float(step_size),
        "noise_scale": float(noise_scale),
        "angle_range": float(angle_range),
        "stroke_width": float(stroke_width),
        "alpha_max": int(alpha_max),
        "alpha_min": int(alpha_min),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "stroke_taper": float(stroke_taper), "stroke_width_var": float(stroke_width_var),
        "stroke_break_density": float(stroke_break_density), "stroke_roughness": float(stroke_roughness),
    }
    label = "streamlines"
elif gen_type == "Voronoi":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_cells": int(n_cells),
        "noise_scale": float(noise_scale),
        "noise_influence": float(noise_influence),
        "filled": filled,
        "stroke_width": float(stroke_w),
        "alpha_min": int(alpha_min), "alpha_max": int(alpha_max),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "bg_hex": bg_hex, "fg_hex": fg_hex,
    }
    label = "voronoi"
elif gen_type == "Cubes":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_cubes": int(n_cubes),
        "size_median": float(size_median), "size_spread": float(size_spread),
        "noise_scale": float(noise_scale), "noise_influence": float(noise_influence),
        "filled": filled, "stroke_width": float(stroke_w),
        "alpha_min": int(alpha_min), "alpha_max": int(alpha_max),
        "shade_contrast": float(shade_contrast),
        "warp_type": warp_type, "warp_amplitude": float(warp_amplitude), "warp_scale": float(warp_scale),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "bg_hex": bg_hex, "fg_hex": fg_hex,
    }
    label = "cubes"
elif gen_type == "Grid":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "cube_size": int(cube_size),
        "max_height": int(max_height),
        "height_style": height_style,
        "height_noise_scale": float(height_noise_scale),
        "noise_scale": float(gr_noise_scale),
        "noise_influence": float(gr_noise_influence),
        "filled": gr_filled, "stroke_width": float(gr_stroke_w),
        "alpha_min": int(gr_alpha_min), "alpha_max": int(gr_alpha_max),
        "shade_contrast": float(gr_shade_contrast),
        "bg_hex": bg_hex, "fg_hex": fg_hex,
    }
    label = "grid"
elif gen_type == "Spirograph":
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "R": int(R_val), "r": int(r_val), "d": float(d_val),
        "mode": "hypo" if spiro_mode.startswith("Hypo") else "epi",
        "n_repeats": int(n_repeats),
        "stroke_width": float(stroke_width),
        "alpha_max": int(sp_alpha_max), "alpha_min": int(sp_alpha_min),
        "margin": float(margin), "gravity": 0.0, "gravity_falloff": 0.0,
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "stroke_taper": float(stroke_taper), "stroke_width_var": float(stroke_width_var),
        "stroke_break_density": float(stroke_break_density), "stroke_roughness": float(stroke_roughness),
    }
    label = "spirograph"
else:
    config = {
        "seed": int(seed),
        "output_width": out_w, "output_height": out_h,
        "n_circles": n_circles,
        "noise_scale": noise_scale, "noise_influence": float(noise_influence),
        "radius_median": float(radius_median), "radius_spread": float(radius_spread),
        "margin": float(margin), "gravity": float(gravity), "gravity_falloff": float(gravity_falloff),
        "filled": filled, "stroke_width": stroke_w,
        "alpha_min": alpha_min, "alpha_max": alpha_max,
        "bg_hex": bg_hex, "fg_hex": fg_hex,
        "invert_overlap": invert_overlap,
    }
    label = "circles"


# ── Cached generation ──────────────────────────────────────────────────────────

@st.cache_data(max_entries=40, show_spinner=False)
def _render(gen_type: str, cfg_key: tuple, scale: float) -> bytes:
    cfg = dict(zip(cfg_key[::2], cfg_key[1::2]))
    fn = (gen_lines.generate           if gen_type == "Lines"
          else gen_circles.generate       if gen_type == "Circles"
          else gen_wave.generate          if gen_type == "Wave"
          else gen_pendulum.generate      if gen_type == "Pendulum"
          else gen_shapes.generate        if gen_type == "Shapes"
          else gen_streamlines.generate   if gen_type == "Streamlines"
          else gen_voronoi.generate       if gen_type == "Voronoi"
          else gen_cubes.generate         if gen_type == "Cubes"
          else gen_cubes_grid.generate    if gen_type == "Grid"
          else gen_spirograph.generate    if gen_type == "Spirograph"
          else gen_attractor.generate)
    img = fn(cfg, scale=scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _cfg_key(cfg: dict) -> tuple:
    return tuple(x for pair in sorted(cfg.items()) for x in pair)


# ── Render preview ─────────────────────────────────────────────────────────────

PREVIEW_MAX_W = 900
preview_scale = min(PREVIEW_MAX_W / out_w, 1.0)
key = _cfg_key(config)

with st.spinner("Rendering..."):
    preview_bytes = _render(gen_type, key, preview_scale)

preview_img = Image.open(io.BytesIO(preview_bytes))
img_placeholder.image(preview_img, use_container_width=True)
gravity_note = f" · gravity {gravity:.2f}" if gravity > 0 else ""
caption_placeholder.caption(
    f"{_MOOD_LABELS[mood]}  ·  {bg_hex} / {fg_hex}  ·  "
    f"preview {preview_img.width}×{preview_img.height} px"
    f"{gravity_note}"
)

# ── Export ─────────────────────────────────────────────────────────────────────

if render_full:
    with st.spinner(f"Rendering {out_w}x{out_h} px..."):
        full_bytes = _render(gen_type, key, 1.0)
    st.download_button(
        "Download PNG",
        data=full_bytes,
        file_name=f"scattered_{label}.png",
        mime="image/png",
    )
