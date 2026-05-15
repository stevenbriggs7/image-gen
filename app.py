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

# Colour seed session state — randomised on first load and on mood change
if "art_seed" not in st.session_state:
    st.session_state.art_seed = 42
if "colour_seed" not in st.session_state:
    st.session_state.colour_seed = 0
if "prev_mood" not in st.session_state:
    st.session_state.prev_mood = None

# Image placeholder at top — filled after params are collected
img_placeholder     = st.empty()
caption_placeholder = st.empty()

# ── Controls panel ─────────────────────────────────────────────────────────────

with st.container(height=500, border=False):

    # Type picker
    gen_type = st.radio(
        "Type", ["Lines", "Circles", "Wave"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # Seed (shared)
    defaults = (gen_lines.DEFAULTS if gen_type == "Lines"
                else gen_circles.DEFAULTS if gen_type == "Circles"
                else gen_wave.DEFAULTS)
    col_btn, col_val = st.columns([2, 3])
    with col_btn:
        if st.button("Randomize", use_container_width=True):
            st.session_state.art_seed = random.randint(0, 99999)
            st.rerun()
    with col_val:
        st.write("")
        st.caption(f"seed {st.session_state.art_seed}")
    seed = st.session_state.art_seed

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

    # ── Composition (shared) ──────────────────────────────────────────────────
    st.subheader("Composition")
    margin = st.slider(
        "Edge margin", 0.0, 0.45, defaults["margin"],
        step=0.01, format="%.2f",
        help="Fraction of the image to leave empty at each edge.",
        key="shared_margin",
    )
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

    # ── Mood / Color & Output (shared) ────────────────────────────────────────
    st.subheader("Mood & Output")
    col_mood, col_reroll = st.columns([3, 1])
    with col_mood:
        mood = st.selectbox(
            "Mood",
            list(_MOOD_LABELS.keys()),
            format_func=_MOOD_LABELS.get,
            key="shared_mood",
        )
    with col_reroll:
        st.write("")
        if st.button("Re-roll", key="shared_reroll", use_container_width=True,
                     help="Pick new colours within this mood"):
            st.session_state.colour_seed = random.randint(0, 99999)
            st.rerun()

    # Randomise colours whenever the mood changes
    if mood != st.session_state.prev_mood:
        st.session_state.colour_seed = random.randint(0, 99999)
        st.session_state.prev_mood = mood

    bg_hex, fg_hex = mood_colors(mood, st.session_state.colour_seed)

    out_w = st.select_slider(
        "Width (px)", [400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3000, 4000],
        value=defaults["output_width"], key="shared_out_w",
    )
    out_h = st.select_slider(
        "Height (px)", [300, 400, 600, 800, 1000, 1200, 1600, 2000, 3000],
        value=defaults["output_height"], key="shared_out_h",
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
    }
    label = "wave"
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
    fn = (gen_lines.generate if gen_type == "Lines"
          else gen_circles.generate if gen_type == "Circles"
          else gen_wave.generate)
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
