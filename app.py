import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append(".")
from preprocessing import preprocess_flux

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kepler Deep Space Observatory",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS + starfield ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

html, body, [class*="st-"], [data-testid] {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}
.stApp {
    background-color: #05070d !important;
}
section[data-testid="stSidebar"] { display: none; }

/* Header */
h1 { color: #00eaff !important; text-shadow: 0 0 20px rgba(0,234,255,0.67), 0 0 40px rgba(0,234,255,0.27); letter-spacing: 6px !important; }
h2, h3 { color: #7b61ff !important; letter-spacing: 3px !important; }
p, span, label, div { color: #a0c0e0 !important; }

/* Tabs */
[data-testid="stTabs"] button {
    background: #080c1a !important;
    color: #a0c0e0 !important;
    border: 1px solid #0f2040 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00eaff !important;
    border-bottom: 2px solid #00eaff !important;
    background: #080c1a !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #05070d, #0a1530) !important;
    border: 1px solid #00eaff !important;
    color: #00eaff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1em !important;
    letter-spacing: 4px !important;
    text-shadow: 0 0 12px rgba(0,234,255,0.73) !important;
    box-shadow: 0 0 18px rgba(0,234,255,0.13) !important;
    border-radius: 4px !important;
    width: 100% !important;
    padding: 0.6em 1em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 40px rgba(0,234,255,0.33) !important;
    border-color: rgba(0,234,255,0.8) !important;
}

/* Sliders */
[data-testid="stSlider"] div[role="slider"] { background: #00eaff !important; }
[data-testid="stSlider"] label { color: #a0c0e0 !important; letter-spacing: 1px; }

/* Radio */
[data-testid="stRadio"] label { color: #a0c0e0 !important; }
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] { border-color: #7b61ff !important; }

/* Metric */
[data-testid="stMetric"] {
    background: #080c1a !important;
    border: 1px solid #0f2040 !important;
    border-radius: 6px !important;
    padding: 12px !important;
}
[data-testid="stMetricValue"] { color: #00eaff !important; font-size: 1.4em !important; }
[data-testid="stMetricLabel"] { color: #a0c0e0 !important; letter-spacing: 2px !important; }

/* Divider */
hr { border-color: #0f2040 !important; }

/* Plotly transparent bg */
.js-plotly-plot, .plot-container { background: transparent !important; }

/* Status bar */
.status-bar { display:flex; gap:24px; padding:6px 2px; font-size:0.8em; }
.status-bar span { color: #00eaff; }
.status-bar span.purple { color: #7b61ff; }

#star-canvas {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none; z-index: 0;
}
.stApp > * { position: relative; z-index: 1; }
</style>

<canvas id="star-canvas"></canvas>
<script>
const c = document.getElementById('star-canvas');
const x = c.getContext('2d');
function resize() { c.width = window.innerWidth; c.height = window.innerHeight; }
resize(); window.addEventListener('resize', resize);
const stars = Array.from({length: 300}, () => ({
    px: Math.random(), py: Math.random(),
    r: Math.random() * 1.2 + 0.1,
    phase: Math.random() * Math.PI * 2,
    speed: 0.004 + Math.random() * 0.008,
    color: Math.random() > 0.85 ? '180,160,255' : '180,210,255'
}));
function draw() {
    x.clearRect(0, 0, c.width, c.height);
    stars.forEach(s => {
        s.phase += s.speed;
        const a = 0.2 + 0.8 * Math.abs(Math.sin(s.phase));
        x.beginPath();
        x.arc(s.px * c.width, s.py * c.height, s.r, 0, Math.PI*2);
        x.fillStyle = 'rgba(' + s.color + ',' + a + ')';
        x.fill();
    });
    requestAnimationFrame(draw);
}
draw();
</script>
""", unsafe_allow_html=True)

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_data
def load_data():
    df = pd.read_csv("demo_stars.csv")
    return df

model   = load_model()
demo_df = load_data()
flux_cols = [c for c in demo_df.columns if c != "LABEL"]

EXOPLANET_FACTS = [
    "Kepler-452b orbits its star in 385 days — the most Earth-like year ever found.",
    "TRAPPIST-1 hosts 7 Earth-sized planets, 3 in the habitable zone.",
    "Hot Jupiters orbit so close to their stars that a year lasts just days.",
    "Kepler-16b orbits two suns — just like Tatooine in Star Wars.",
    "Over 5,500 exoplanets have been confirmed. Billions more are estimated to exist.",
    "Some exoplanets rain molten iron on their night sides.",
    "HD 209458 b was the first exoplanet caught transiting its star — in 1999.",
    "Proxima Centauri b is the closest known exoplanet, just 4.2 light-years away.",
]

PLOT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#05070d",
    plot_bgcolor="#080c1a",
    font=dict(color="#a0c0e0", family="Share Tech Mono, Courier New"),
    margin=dict(t=44, b=36, l=52, r=16),
)

# ── Plot helpers ──────────────────────────────────────────────────────────────
def make_gauge(prob):
    pct   = round(prob * 100, 1)
    color = "#00eaff" if prob > 0.5 else "#ff4466"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(color=color, size=34, family="Share Tech Mono")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#3a5080",
                      tickfont=dict(color="#3a5080", family="Share Tech Mono"), nticks=6),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#080c1a", borderwidth=1, bordercolor="#0f2040",
            steps=[
                dict(range=[0,  50],  color="rgba(255,68,102,0.07)"),
                dict(range=[50, 100], color="rgba(0,234,255,0.07)"),
            ],
            threshold=dict(line=dict(color="rgba(255,255,255,0.33)", width=1), thickness=0.7, value=50),
        ),
        title=dict(text="TRANSIT PROBABILITY", font=dict(color="#a0c0e0", size=11, family="Share Tech Mono"))
    ))
    fig.update_layout(**PLOT_BASE, height=240)
    return fig

def make_sensor_feed(flux_raw, flux_processed, star_id):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("RAW PHOTOMETRIC SIGNAL", "PROCESSED SIGNAL  (CNN INPUT)"),
                        horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(y=flux_raw, mode="lines",
                             line=dict(color="#00eaff", width=1), name="RAW"), row=1, col=1)
    fig.add_trace(go.Scatter(y=flux_processed, mode="lines",
                             line=dict(color="#7b61ff", width=1), name="PROCESSED"), row=1, col=2)
    fig.update_layout(**PLOT_BASE, height=300, showlegend=False,
                      title=dict(text="PHOTOMETRIC SENSOR FEED  ·  STAR #" + str(star_id).zfill(3),
                                 font=dict(color="#a0c0e0", size=12)))
    fig.update_xaxes(title_text="OBSERVATION INDEX", gridcolor="#0f2040", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(title_text="FLUX", gridcolor="#0f2040", zeroline=False, tickfont=dict(size=9))
    fig.update_annotations(font=dict(color="#7b61ff", size=11))
    return fig

def make_period_plots(flux_raw, t0, period_len, star_id):
    n      = len(flux_raw)
    t0     = int(np.clip(t0, 0, n - 1))
    t1     = int(np.clip(t0 + period_len, 1, n))
    period = flux_raw[t0:t1]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=flux_raw, mode="lines",
                              line=dict(color="#00eaff", width=1), name="Light Curve"))
    fig1.add_vrect(x0=t0, x1=t1, fillcolor="rgba(123,97,255,0.15)",
                   line=dict(color="#7b61ff", width=1),
                   annotation_text="TRANSIT WINDOW",
                   annotation_font=dict(color="#7b61ff", size=10),
                   annotation_position="top left")
    fig1.update_layout(
        **PLOT_BASE, height=300, showlegend=False,
        title=dict(text="FULL LIGHT CURVE  ·  STAR #" + str(star_id).zfill(3) +
                        "  ·  Window: t0=" + str(t0) + "  len=" + str(period_len),
                   font=dict(color="#a0c0e0", size=12)),
        xaxis=dict(title="OBSERVATION POINT", gridcolor="#0f2040", zeroline=False),
        yaxis=dict(title="NORMALIZED FLUX",   gridcolor="#0f2040", zeroline=False),
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=period, mode="lines",
                              line=dict(color="#7b61ff", width=1.5), name="Selected Period"))
    fig2.add_hline(y=0, line=dict(color="rgba(160,192,224,0.2)", width=1, dash="dot"))
    fig2.update_layout(
        **PLOT_BASE, height=260, showlegend=False,
        title=dict(text="ZOOMED TRANSIT WINDOW  ·  " + str(len(period)) + " observations",
                   font=dict(color="#a0c0e0", size=12)),
        xaxis=dict(title="OBSERVATION POINT", gridcolor="#0f2040", zeroline=False),
        yaxis=dict(title="NORMALIZED FLUX",   gridcolor="#0f2040", zeroline=False),
    )
    return fig1, fig2

def compute_telemetry(flux_raw):
    noise     = float(np.std(flux_raw))
    mn        = float(np.mean(flux_raw))
    snr       = abs(mn / noise) if noise > 0 else 0
    depth_pct = float((np.max(flux_raw) - np.min(flux_raw)) / (abs(mn) + 1e-9) * 100)
    return round(snr, 2), round(depth_pct, 2), round(noise, 2)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⬡ KEPLER DEEP SPACE OBSERVATORY")
st.markdown("### EXOPLANET DETECTION SYSTEM  ·  NEURAL TRANSIT CLASSIFIER v1.0")
st.markdown("""
<div class="status-bar">
    <span>● SYSTEM ONLINE</span>
    <span>● MODEL READY</span>
    <span class="purple">● KEPLER DATASET LOADED</span>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔭  NEURAL TRANSIT SCAN", "📡  PERIOD EXPLORER"])

# ── Tab 1: Neural Scan ────────────────────────────────────────────────────────
with tab1:
    col_slider, col_guess = st.columns([3, 1])
    with col_slider:
        star_index = st.slider("STAR SYSTEM SELECTOR  ·  Target ID [0–199]", 0, 199, 0, key="scan_star")
    with col_guess:
        user_guess = st.radio(
            "MISSION PREDICTION  ·  Operator Hypothesis",
            ["🌍 Exoplanet", "⭐ Not an exoplanet", "🤔 Skip"],
            index=2, key="guess"
        )

    scan_btn = st.button("🔭  INITIATE TRANSIT SCAN", key="scan_btn")

    # Compute on load or button press
    if scan_btn or "scan_result" not in st.session_state:
        row      = demo_df.iloc[int(star_index)]
        flux_raw = row[flux_cols].values.astype(float)
        processed = preprocess_flux(flux_raw)
        prob     = float(model.predict(processed, verbose=0)[0][0])
        label    = int(row["LABEL"])
        planet   = prob > 0.5
        correct  = planet == (label == 1)
        snr, depth, noise = compute_telemetry(flux_raw)

        st.session_state.scan_result = dict(
            feed=make_sensor_feed(flux_raw, processed.flatten(), star_index),
            gauge=make_gauge(prob),
            planet=planet, prob=prob, label=label,
            correct=correct, snr=snr, depth=depth, noise=noise,
            star_index=star_index, user_guess=user_guess,
        )

    r = st.session_state.scan_result
    st.plotly_chart(r["feed"], use_container_width=True, key="scan_feed")

    col_gauge, col_result = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(r["gauge"], use_container_width=True, key="scan_gauge")
    with col_result:
        verdict  = "## 🟢 TRANSIT SIGNAL DETECTED" if r["planet"] else "## 🔴 NO TRANSIT SIGNAL"
        true_str = "CONFIRMED EXOPLANET" if r["label"] == 1 else "NON-PLANETARY STAR"
        correct_str = "✅ PREDICTION CORRECT" if r["correct"] else "❌ PREDICTION INCORRECT"
        st.markdown(verdict)
        st.markdown(f"**CATALOG STATUS:** {true_str}   ·   {correct_str}")
        st.markdown(f"**TELEMETRY ·**  SNR: {r['snr']}  ·  Depth: {r['depth']}%  ·  Noise: {r['noise']}")

        if r["user_guess"] != "🤔 Skip":
            guessed = r["user_guess"] == "🌍 Exoplanet"
            ok      = guessed == (r["label"] == 1)
            st.markdown(f"**OPERATOR HYPOTHESIS:** {'✅ CONFIRMED' if ok else '❌ REFUTED'}")

        if r["planet"]:
            st.info("💡 " + EXOPLANET_FACTS[r["star_index"] % len(EXOPLANET_FACTS)])

# ── Tab 2: Period Explorer ────────────────────────────────────────────────────
with tab2:
    st.markdown("### MANUAL TRANSIT PERIOD ANALYSIS")
    st.markdown(
        "Adjust the **transit window** to isolate the dip in the light curve. "
        "When you've found it, the CNN score should jump above 50%."
    )

    p_star = st.slider("STAR SYSTEM SELECTOR  ·  Target ID [0–199]", 0, 199, 0, key="period_star")
    col_t0, col_len = st.columns(2)
    with col_t0:
        t0 = st.slider("TRANSIT START  ·  t₀  (observation index)", 0, 3000, 400, step=10)
    with col_len:
        period_len = st.slider("PERIOD LENGTH  ·  number of observations", 10, 2000, 800, step=10)

    p_btn = st.button("📡  ANALYZE TRANSIT WINDOW", key="period_btn")

    if p_btn or "period_result" not in st.session_state:
        row      = demo_df.iloc[int(p_star)]
        flux_raw = row[flux_cols].values.astype(float)
        n        = len(flux_raw)
        t0c      = int(np.clip(t0, 0, n - 1))
        plenc    = int(np.clip(period_len, 10, n - t0c))

        fig1, fig2 = make_period_plots(flux_raw, t0c, plenc, p_star)

        window = flux_raw[t0c : t0c + plenc]
        padded = np.zeros_like(flux_raw)
        padded[:len(window)] = window
        processed = preprocess_flux(padded)
        prob      = float(model.predict(processed, verbose=0)[0][0])
        label     = int(row["LABEL"])
        snr, depth, noise = compute_telemetry(window)

        st.session_state.period_result = dict(
            fig1=fig1, fig2=fig2, gauge=make_gauge(prob),
            prob=prob, label=label, snr=snr, depth=depth, noise=noise,
        )

    pr = st.session_state.period_result
    st.plotly_chart(pr["fig1"], use_container_width=True, key="period_fig1")
    st.plotly_chart(pr["fig2"], use_container_width=True, key="period_fig2")

    col_pgauge, col_presult = st.columns([1, 2])
    with col_pgauge:
        st.plotly_chart(pr["gauge"], use_container_width=True, key="period_gauge")
    with col_presult:
        verdict  = "## 🟢 TRANSIT DETECTED IN WINDOW" if pr["prob"] > 0.5 else "## 🔴 NO TRANSIT IN WINDOW"
        true_str = "CONFIRMED EXOPLANET" if pr["label"] == 1 else "NON-PLANETARY STAR"
        st.markdown(verdict)
        st.markdown(f"**WINDOW PROBABILITY:** {round(pr['prob'] * 100, 1)}%")
        st.markdown(f"**CATALOG LABEL:** {true_str}")
        st.markdown(f"**WINDOW TELEMETRY ·**  SNR: {pr['snr']}  ·  Depth: {pr['depth']}%  ·  Noise: {pr['noise']}")
        st.caption("Adjust t0 and period length to isolate the transit dip, then compare with the CNN score above.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("KEPLER SPACE TELESCOPE DATA  ·  1D CNN  ·  Fourier → Savitzky-Golay → Normalization → Robust Scaling")
