"""
Streamlit UI -- generative image tools.

Run:
    streamlit run app.py
"""

import io
import random
import streamlit as st
from PIL import Image

import generate as gen_lines
import generate_circles as gen_circles

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

# Image placeholder at top — filled after params are collected
img_placeholder     = st.empty()
caption_placeholder = st.empty()

# ── Controls panel ─────────────────────────────────────────────────────────────

with st.container(height=500, border=False):

    # Type picker
    gen_type = st.radio(
        "Type", ["Lines", "Circles"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # Seed (shared)
    defaults = gen_lines.DEFAULTS if gen_type == "Lines" else gen_circles.DEFAULTS
    col_seed, col_btn = st.columns([5, 1])
    with col_seed:
        seed = st.number_input("Seed", min_value=0, max_value=99999,
                               value=defaults["seed"], step=1)
    with col_btn:
        st.write("")
        if st.button("Randomize", help="Randomize seed", use_container_width=True):
            seed = random.randint(0, 99999)
            st.rerun()

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
    else:
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
    # ── Color & Output (shared) ───────────────────────────────────────────────
    st.subheader("Color & Output")
    col_bg, col_fg = st.columns(2)
    with col_bg:
        bg_hex = st.color_picker("Background", defaults["bg_hex"], key="shared_bg")
    with col_fg:
        fg_hex = st.color_picker("Mark color", defaults["fg_hex"], key="shared_fg")
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
    fn = gen_lines.generate if gen_type == "Lines" else gen_circles.generate
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
    f"Preview {preview_img.width}x{preview_img.height} px "
    f"- output {out_w}x{out_h} px{gravity_note}"
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
