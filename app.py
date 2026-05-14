"""
Streamlit UI for the scattered-lines generator.

Run:
    streamlit run app.py
"""

import io
import random
import streamlit as st
from PIL import Image

from generate import generate, DEFAULTS

st.set_page_config(
    page_title="Scattered Lines",
    page_icon="╱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Scattered Lines Generator")

# ── Sidebar controls ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")

    # -- Randomness --
    st.subheader("Randomness")
    col_seed, col_btn = st.columns([3, 1])
    with col_seed:
        seed = st.number_input("Seed", min_value=0, max_value=99999,
                               value=DEFAULTS["seed"], step=1, label_visibility="collapsed")
    with col_btn:
        if st.button("⟳", help="Randomize seed"):
            seed = random.randint(0, 99999)
            st.rerun()

    st.caption(f"seed: {seed}")

    # -- Lines --
    st.subheader("Lines")
    n_lines = st.slider("Count", 50, 5000, DEFAULTS["n_lines"], step=50)
    length_mu = st.slider("Length mean (px)", 10, 400, int(DEFAULTS["length_mu"]))
    length_sigma = st.slider("Length std dev", 1, 200, int(DEFAULTS["length_sigma"]))

    # -- Flow field --
    st.subheader("Flow Field")
    noise_scale = st.slider(
        "Noise scale", 0.0005, 0.025, DEFAULTS["noise_scale"],
        step=0.0005, format="%.4f",
        help="Low = sweeping coherent directions. High = chaotic.",
    )
    angle_range = st.slider(
        "Angle range (× 2π)", 0.05, 1.0, DEFAULTS["angle_range"],
        step=0.05, format="%.2f",
        help="1.0 = lines can point in any direction; 0.25 = constrained to a quadrant.",
    )

    # -- Stroke --
    st.subheader("Stroke")
    sw_min, sw_max = st.slider(
        "Width range (px)", 0.5, 8.0,
        (DEFAULTS["stroke_width_min"], DEFAULTS["stroke_width_max"]),
        step=0.5,
    )
    alpha_min, alpha_max = st.slider(
        "Opacity range (0–255)", 0, 255,
        (DEFAULTS["alpha_min"], DEFAULTS["alpha_max"]),
    )

    # -- Color --
    st.subheader("Color")
    bg_dark = st.toggle("Dark background", value=DEFAULTS["bg_dark"])

    # -- Output size --
    st.subheader("Output")
    out_w = st.select_slider(
        "Width (px)", [400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3000, 4000],
        value=DEFAULTS["output_width"],
    )
    out_h = st.select_slider(
        "Height (px)", [300, 400, 600, 800, 1000, 1200, 1600, 2000, 3000],
        value=DEFAULTS["output_height"],
    )

# ── Assemble config ────────────────────────────────────────────────────────────

config = {
    "seed": int(seed),
    "output_width": out_w,
    "output_height": out_h,
    "n_lines": n_lines,
    "noise_scale": noise_scale,
    "angle_range": angle_range,
    "length_mu": float(length_mu),
    "length_sigma": float(length_sigma),
    "stroke_width_min": sw_min,
    "stroke_width_max": sw_max,
    "alpha_min": alpha_min,
    "alpha_max": alpha_max,
    "bg_dark": bg_dark,
}


# ── Cached generation (PNG bytes, so Streamlit can serialise it) ───────────────

@st.cache_data(max_entries=30, show_spinner=False)
def _generate_preview(cfg_key: tuple, scale: float) -> bytes:
    cfg = dict(zip(cfg_key[::2], cfg_key[1::2]))
    img = generate(cfg, scale=scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


@st.cache_data(max_entries=5, show_spinner=False)
def _generate_full(cfg_key: tuple) -> bytes:
    cfg = dict(zip(cfg_key[::2], cfg_key[1::2]))
    img = generate(cfg, scale=1.0)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _cfg_key(cfg: dict) -> tuple:
    return tuple(x for pair in sorted(cfg.items()) for x in pair)


# ── Preview ────────────────────────────────────────────────────────────────────

# Scale preview so it fits comfortably on screen without rounding output size
PREVIEW_MAX_W = 900
preview_scale = min(PREVIEW_MAX_W / out_w, 1.0)

key = _cfg_key(config)

with st.spinner("Rendering preview…"):
    preview_bytes = _generate_preview(key, scale=preview_scale)

preview_img = Image.open(io.BytesIO(preview_bytes))
st.image(preview_img, use_container_width=True)

st.caption(
    f"Preview at {preview_img.width}×{preview_img.height} px "
    f"(output: {out_w}×{out_h} px, {n_lines:,} lines)"
)

# ── Download full resolution ───────────────────────────────────────────────────

col_dl, col_info = st.columns([1, 3])

with col_dl:
    if st.button("Render full resolution", type="primary"):
        with st.spinner(f"Rendering {out_w}×{out_h} px…"):
            full_bytes = _generate_full(key)
        st.download_button(
            "⬇ Download PNG",
            data=full_bytes,
            file_name="scattered_lines.png",
            mime="image/png",
        )

with col_info:
    st.markdown(
        "**Tip:** adjust any parameter and the preview updates automatically. "
        "Click *Render full resolution* to export at the chosen output size."
    )
