import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ==============================
# CUSTOM CSS - DARK TECH THEME
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f1629;
    --bg-card: #141d35;
    --bg-card-hover: #1a2540;
    --accent-cyan: #00e5ff;
    --accent-violet: #7b61ff;
    --accent-green: #00e676;
    --accent-amber: #ffab40;
    --accent-red: #ff5252;
    --text-primary: #e8eaf6;
    --text-secondary: #8892b0;
    --text-muted: #4a5568;
    --border: rgba(0, 229, 255, 0.12);
    --border-hover: rgba(0, 229, 255, 0.35);
    --glow-cyan: 0 0 20px rgba(0, 229, 255, 0.25);
    --glow-violet: 0 0 20px rgba(123, 97, 255, 0.25);
}

/* ---- GLOBAL RESET ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ---- ANIMATED BACKGROUND GRID ---- */
.stApp {
    background-color: var(--bg-primary) !important;
    background-image:
        linear-gradient(rgba(0, 229, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 229, 255, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    min-height: 100vh;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1020 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em;
    padding: 6px 0 !important;
    transition: color 0.2s;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent-cyan) !important;
}

section[data-testid="stSidebar"] h1 {
    color: var(--accent-cyan) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
}

/* ---- HEADINGS ---- */
h1, h2, h3, h4 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}

h1 {
    font-size: 2rem !important;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2em !important;
}

h3 {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1.05rem !important;
}

/* ---- METRIC / INFO CARDS ---- */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
    transition: border-color 0.3s, box-shadow 0.3s;
}

.info-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--glow-cyan);
}

.badge {
    display: inline-block;
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.25);
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 6px;
}

/* ---- INPUTS & SLIDERS ---- */
.stNumberInput input, .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Slider thumb */
.stSlider > div > div > div > div {
    background: var(--accent-cyan) !important;
}

.stSlider > div > div > div {
    background: rgba(0, 229, 255, 0.15) !important;
}

/* ---- LABELS ---- */
label, .stSlider label, .stSelectbox label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ---- BUTTON ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(0, 229, 255, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(0, 229, 255, 0.4) !important;
    filter: brightness(1.08) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ---- DATAFRAME ---- */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ---- DIVIDER ---- */
hr {
    border-color: var(--border) !important;
    margin: 24px 0 !important;
}

/* ---- SECTION LABEL ---- */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 4px;
    opacity: 0.8;
}

/* ---- PREDICTION RESULT ---- */
.prediction-box {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    position: relative;
    overflow: hidden;
}

.prediction-box::before {
    content: '';
    position: absolute;
    inset: 0;
    background: inherit;
    filter: blur(40px);
    opacity: 0.4;
    z-index: -1;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 3px; }

/* ---- WARNINGS ---- */
.stAlert {
    background: rgba(255, 171, 64, 0.08) !important;
    border: 1px solid rgba(255, 171, 64, 0.3) !important;
    border-radius: 10px !important;
    color: var(--accent-amber) !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD MODEL & SCALER
# ==============================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("mobile_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model_and_scaler()


# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        return df
    except:
        return None


df = load_data()


# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "",
    ["Home", "Data Overview", "Price Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="color:#4a5568; font-size:0.75rem; font-family: Space Mono, monospace; letter-spacing:0.05em;">'
    'MOBILE PRICE AI v1.0<br>Powered by Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)


# ==============================
# HOME PAGE
# ==============================
if page == "Home":
    st.markdown('<div class="section-label">Overview</div>',
                unsafe_allow_html=True)
    st.title("📱 Mobile Price Predictor")
    st.markdown(
        '<p style="color:#8892b0; font-size:1.05rem; margin-bottom:32px;">'
        'Classify any smartphone into a price tier using its hardware specifications — instantly.'
        '</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="section-label">Model</div>
            <h2 style="font-family: Space Mono, monospace; font-size:1.1rem; color:#e8eaf6; margin-top:8px;">
                Logistic Regression
            </h2>
            <p style="color:#8892b0; font-size:0.9rem; line-height:1.7;">
                Trained with <strong style="color:#00e5ff;">StandardScaler</strong> normalization.
                Multi-class classification across 4 price tiers.
                20 hardware features as input.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="section-label">Price Tiers</div>
            <div style="margin-top: 12px; display: flex; flex-direction: column; gap: 8px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="width:10px; height:10px; border-radius:50%; background:#00e676; display:inline-block; flex-shrink:0;"></span>
                    <span style="color:#8892b0; font-size:0.9rem;"><strong style="color:#e8eaf6;">0</strong> — Low Cost</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="width:10px; height:10px; border-radius:50%; background:#00e5ff; display:inline-block; flex-shrink:0;"></span>
                    <span style="color:#8892b0; font-size:0.9rem;"><strong style="color:#e8eaf6;">1</strong> — Medium Cost</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="width:10px; height:10px; border-radius:50%; background:#ffab40; display:inline-block; flex-shrink:0;"></span>
                    <span style="color:#8892b0; font-size:0.9rem;"><strong style="color:#e8eaf6;">2</strong> — High Cost</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="width:10px; height:10px; border-radius:50%; background:#7b61ff; display:inline-block; flex-shrink:0;"></span>
                    <span style="color:#8892b0; font-size:0.9rem;"><strong style="color:#e8eaf6;">3</strong> — Very High Cost</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="color:#4a5568; font-size:0.85rem;">'
        '👉 Navigate to <strong style="color:#00e5ff;">Price Prediction</strong> in the sidebar to test the model.'
        '</p>',
        unsafe_allow_html=True
    )


# ==============================
# DATA OVERVIEW PAGE
# ==============================
elif page == "Data Overview":
    st.markdown('<div class="section-label">Dataset</div>',
                unsafe_allow_html=True)
    st.title("Data Overview")

    if df is not None:
        col_a, col_b, col_c = st.columns(3, gap="medium")
        with col_a:
            st.markdown(f"""
            <div class="info-card" style="text-align:center;">
                <div class="section-label">Rows</div>
                <div style="font-family: Space Mono, monospace; font-size:2rem; color:#00e5ff; margin-top:6px;">
                    {df.shape[0]:,}
                </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="info-card" style="text-align:center;">
                <div class="section-label">Columns</div>
                <div style="font-family: Space Mono, monospace; font-size:2rem; color:#7b61ff; margin-top:6px;">
                    {df.shape[1]}
                </div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="info-card" style="text-align:center;">
                <div class="section-label">Missing Values</div>
                <div style="font-family: Space Mono, monospace; font-size:2rem; color:#00e676; margin-top:6px;">
                    {df.isnull().sum().sum()}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Preview</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("---")
        st.markdown(
            '<div class="section-label">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("⚠️  dataset.csv not found in project folder.")


# ==============================
# PREDICTION PAGE
# ==============================
elif page == "Price Prediction":
    st.markdown('<div class="section-label">Inference</div>',
                unsafe_allow_html=True)
    st.title("Price Prediction")
    st.markdown(
        '<p style="color:#8892b0; font-size:0.95rem; margin-bottom:28px;">'
        'Fill in the hardware specs below, then hit Predict to classify the device.'
        '</p>',
        unsafe_allow_html=True
    )

    yes_no_options = ["No", "Yes"]

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            '<div class="section-label" style="margin-bottom:12px;">Core & Connectivity</div>', unsafe_allow_html=True)
        battery_power = st.number_input("Battery Power (mAh)", 500, 3000, 1500)
        blue = st.selectbox("Bluetooth", yes_no_options)
        blue = 1 if blue == "Yes" else 0
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
        dual_sim = st.selectbox("Dual SIM", yes_no_options)
        dual_sim = 1 if dual_sim == "Yes" else 0
        fc = st.slider("Front Camera (MP)", 0, 20, 5)
        four_g = st.selectbox("4G", yes_no_options)
        four_g = 1 if four_g == "Yes" else 0

    with col2:
        st.markdown(
            '<div class="section-label" style="margin-bottom:12px;">Hardware & Build</div>', unsafe_allow_html=True)
        int_memory = st.slider("Internal Memory (GB)", 2, 128, 32)
        m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5)
        mobile_wt = st.slider("Mobile Weight (g)", 80, 250, 150)
        n_cores = st.slider("Number of Cores", 1, 8, 4)
        pc = st.slider("Primary Camera (MP)", 0, 64, 12)
        px_height = st.number_input("Pixel Height", 0, 2000, 800)

    with col3:
        st.markdown(
            '<div class="section-label" style="margin-bottom:12px;">Display & Features</div>', unsafe_allow_html=True)
        px_width = st.number_input("Pixel Width", 0, 2000, 1200)
        ram = st.number_input("RAM (MB)", 256, 8000, 2000)
        sc_h = st.slider("Screen Height (cm)", 5, 25, 12)
        sc_w = st.slider("Screen Width (cm)", 0, 20, 7)
        talk_time = st.slider("Talk Time (hrs)", 2, 30, 10)
        three_g = st.selectbox("3G", yes_no_options)
        three_g = 1 if three_g == "Yes" else 0
        touch_screen = st.selectbox("Touch Screen", yes_no_options)
        touch_screen = 1 if touch_screen == "Yes" else 0
        wifi = st.selectbox("WiFi", yes_no_options)
        wifi = 1 if wifi == "Yes" else 0

    st.markdown("---")

    if st.button("🚀 Predict Price Range", use_container_width=True):
        try:
            features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                                  int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                                  px_width, ram, sc_h, sc_w, talk_time, three_g,
                                  touch_screen, wifi]])

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            price_styles = {
                0: {
                    "label": "Low Cost",
                    "bg": "linear-gradient(135deg, #00401a, #00e676)",
                    "border": "#00e676",
                    "glow": "rgba(0, 230, 118, 0.35)",
                    "tier": "Tier 0"
                },
                1: {
                    "label": "Medium Cost",
                    "bg": "linear-gradient(135deg, #003d4d, #00e5ff)",
                    "border": "#00e5ff",
                    "glow": "rgba(0, 229, 255, 0.35)",
                    "tier": "Tier 1"
                },
                2: {
                    "label": "High Cost",
                    "bg": "linear-gradient(135deg, #3d2800, #ffab40)",
                    "border": "#ffab40",
                    "glow": "rgba(255, 171, 64, 0.35)",
                    "tier": "Tier 2"
                },
                3: {
                    "label": "Very High Cost",
                    "bg": "linear-gradient(135deg, #1a0050, #7b61ff)",
                    "border": "#7b61ff",
                    "glow": "rgba(123, 97, 255, 0.35)",
                    "tier": "Tier 3"
                },
            }

            s = price_styles[prediction]

            st.markdown(f"""
            <div style="
                background: {s['bg']};
                border: 1.5px solid {s['border']};
                border-radius: 16px;
                padding: 36px 28px;
                text-align: center;
                box-shadow: 0 0 48px {s['glow']};
                margin-top: 12px;
            ">
                <div style="font-family: Space Mono, monospace; font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(255,255,255,0.55); margin-bottom: 10px;">
                    Predicted Price Range &nbsp;·&nbsp; {s['tier']}
                </div>
                <div style="font-family: Space Mono, monospace; font-size: 2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">
                    {s['label']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
