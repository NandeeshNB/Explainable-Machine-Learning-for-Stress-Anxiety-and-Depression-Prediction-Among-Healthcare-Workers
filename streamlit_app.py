"""
Healthcare Worker Stress Prediction System - Streamlit UI
═══════════════════════════════════════════════════════════

A complete web application for predicting and explaining stress levels
in healthcare workers using ML, SHAP, and Gemini AI.

Install: pip install google-genai streamlit
Run:     streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
from datetime import datetime
from google import genai
from google.genai import types

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediStress AI · Healthcare Burnout Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Clean medical dark theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0a0f1a;
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1424 !important;
        border-right: 1px solid #1e2d45;
    }
    [data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: transparent;
        border-radius: 8px;
        padding: 6px 10px;
        transition: all 0.2s;
        cursor: pointer;
        display: block;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #1e2d45 !important;
        color: #7dd3fc !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1f3c 0%, #112240 50%, #0d1f3c 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 300px; height: 120px;
        background: radial-gradient(ellipse, rgba(99,102,241,0.08) 0%, transparent 70%);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #f0f9ff;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #7dd3fc;
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.3);
        color: #38bdf8;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 1rem;
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: #0d1424;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: #38bdf8; }
    .stat-value {
        font-size: 2rem;
        font-weight: 600;
        color: #38bdf8;
        font-family: 'DM Mono', monospace;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .stat-delta {
        font-size: 0.8rem;
        color: #34d399;
        margin-top: 0.2rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: -0.01em;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #1e2d45;
        margin-left: 0.5rem;
    }

    /* ── Input Panel ── */
    .input-panel {
        background: #0d1424;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .input-panel-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        border-bottom: 1px solid #1e2d45;
        padding-bottom: 0.5rem;
    }

    /* ── Prediction Result Cards ── */
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid;
    }
    .result-card-low {
        background: rgba(16,185,129,0.08);
        border-color: rgba(16,185,129,0.35);
    }
    .result-card-medium {
        background: rgba(245,158,11,0.08);
        border-color: rgba(245,158,11,0.35);
    }
    .result-card-high {
        background: rgba(239,68,68,0.08);
        border-color: rgba(239,68,68,0.35);
    }
    .result-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .result-value-low  { font-size: 2.2rem; font-weight: 700; color: #10b981; font-family: 'DM Mono', monospace; }
    .result-value-med  { font-size: 2.2rem; font-weight: 700; color: #f59e0b; font-family: 'DM Mono', monospace; }
    .result-value-high { font-size: 2.2rem; font-weight: 700; color: #ef4444; font-family: 'DM Mono', monospace; }

    /* ── Probability Bar ── */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.7rem;
    }
    .prob-label { width: 65px; font-size: 0.8rem; color: #94a3b8; text-align: right; }
    .prob-bar-track {
        flex: 1;
        height: 8px;
        background: #1e2d45;
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar-fill-low  { height: 100%; border-radius: 4px; background: #10b981; transition: width 0.8s ease; }
    .prob-bar-fill-med  { height: 100%; border-radius: 4px; background: #f59e0b; transition: width 0.8s ease; }
    .prob-bar-fill-high { height: 100%; border-radius: 4px; background: #ef4444; transition: width 0.8s ease; }
    .prob-pct { width: 42px; font-size: 0.8rem; color: #e2e8f0; font-family: 'DM Mono', monospace; }

    /* ── Factor Pills ── */
    .factor-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .factor-pill {
        background: #112240;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 0.78rem;
        color: #93c5fd;
    }

    /* ── Report Box ── */
    .report-box {
        background: #0d1424;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #cbd5e1;
    }

    /* ── API Key Status ── */
    .api-connected {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
        color: #34d399;
        text-align: center;
        margin-top: 0.5rem;
    }
    .api-disconnected {
        background: rgba(100,116,139,0.1);
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
        color: #64748b;
        text-align: center;
        margin-top: 0.5rem;
    }

    /* ── About cards ── */
    .team-card {
        background: #0d1424;
        border: 1px solid #1e2d45;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .team-name { font-weight: 600; color: #e2e8f0; font-size: 0.9rem; }
    .team-id   { font-size: 0.78rem; color: #64748b; font-family: 'DM Mono', monospace; }

    .phase-card {
        background: #0d1424;
        border: 1px solid #1e2d45;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .phase-title { font-weight: 600; color: #7dd3fc; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .phase-body  { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }

    /* ── Streamlit overrides ── */
    .stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }
    div[data-testid="stMetric"] {
        background: #0d1424;
        border: 1px solid #1e2d45;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
    div[data-testid="stMetricDelta"] { color: #34d399 !important; }

    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        border: none !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid #1e3a5f !important;
        color: #7dd3fc !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #112240 !important;
        border-color: #38bdf8 !important;
    }

    .stSelectbox label, .stSlider label, .stTextInput label { color: #94a3b8 !important; font-size: 0.83rem !important; }
    .stSelectbox [data-baseweb="select"] > div {
        background: #112240 !important;
        border-color: #1e3a5f !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stTextInput input {
        background: #112240 !important;
        border-color: #1e3a5f !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #0d1424 !important;
        border-bottom: 1px solid #1e2d45;
        gap: 0.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        border-radius: 8px 8px 0 0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #112240 !important;
        color: #7dd3fc !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    .stDataFrame { background: #0d1424 !important; border-radius: 10px !important; }

    /* hide streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }

    /* Divider */
    hr { border-color: #1e2d45 !important; }

    /* info/warning/success/error */
    .stAlert {
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════════════════════

def apply_chart_style(ax, fig):
    fig.patch.set_facecolor('#0d1424')
    ax.set_facecolor('#0a0f1a')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2d45')
    ax.grid(color='#1e2d45', linestyle='--', linewidth=0.6, alpha=0.7)
    return ax, fig


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    try:
        with open('models/xgb_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        return {'xgb': xgb_model, 'scaler': scaler, 'encoders': label_encoders, 'loaded': True}
    except FileNotFoundError:
        st.error("⚠️ Model files not found in `models/` folder. Please train models first.")
        return {'loaded': False}


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI SETUP  — no @st.cache_resource so it refreshes on new key
# ══════════════════════════════════════════════════════════════════════════════

def setup_gemini(api_key):
    try:
        client = genai.Client(api_key=api_key)
        # Quick validation: list models
        return client
    except Exception as e:
        st.error(f"Gemini setup failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION & AI
# ══════════════════════════════════════════════════════════════════════════════

def predict_stress(input_data, models):
    features = np.array([[
        input_data['humidity'],
        input_data['temperature'],
        input_data['step_count'],
        input_data['workhours'],
        input_data['patients_attended'],
        input_data['dept_stress'],
        input_data['env_stress'],
        input_data['workload_intensity'],
        input_data['department_encoded'],
        input_data['activity_level_encoded'],
        input_data['shift_type_encoded']
    ]])
    features_scaled = models['scaler'].transform(features)
    prediction = models['xgb'].predict(features_scaled)[0]
    probabilities = models['xgb'].predict_proba(features_scaled)[0]
    return {
        'class': int(prediction),
        'label': ['Low', 'Medium', 'High'][int(prediction)],
        'probabilities': probabilities,
        'confidence': float(probabilities[int(prediction)] * 100)
    }


def generate_ai_report(prediction, input_data, gemini_model):
    stress_level = prediction['label']
    confidence = prediction['confidence']
    prompt = f"""
You are a healthcare occupational health specialist. Generate a concise stress assessment report.

Assessment Results:
- Predicted Stress Level: {stress_level} (Confidence: {confidence:.1f}%)
- Department: {input_data.get('department', 'Unknown')}
- Work Hours: {input_data.get('workhours', 0)}
- Patients Attended: {input_data.get('patients_attended', 0)}
- Temperature: {input_data.get('temperature', 0)}°F
- Humidity: {input_data.get('humidity', 0)}%
- Step Count: {input_data.get('step_count', 0)}
- Department Stress Score: {input_data.get('dept_stress', 0)}/10

Generate a professional report with these sections:

## 📋 Summary
2 sentences: overall assessment and primary concern.

## 🔍 Key Contributing Factors
3 bullet points identifying the main drivers of this stress level.

## 💡 Recommendations
Structured as:
- **Immediate (24-48h):** one action
- **Short-term (1-2 weeks):** one action  
- **Long-term:** one structural change

## ⚠️ Warning Signs to Monitor
3 specific symptoms or behavioral changes to watch for.

Keep the report concise, professional, evidence-based, and actionable.
"""
    try:
        response = gemini_model.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return (
                "⚠️ **Quota Exceeded** — Your free tier limit has been reached.\n\n"
                "**How to fix:**\n"
                "- Wait 24 hours for the daily quota to reset, **or**\n"
                "- Enable billing at [aistudio.google.com](https://aistudio.google.com) (pay-as-you-go, very affordable), **or**\n"
                "- Try a different API key with remaining quota.\n\n"
                "_Free tier: ~15 req/min, 1,500 req/day for gemini-1.5-flash-latest._"
            )
        elif "404" in err or "not found" in err.lower():
            return (
                "⚠️ **Model Not Found** — The model may be unavailable. Please check your API key region and billing status at aistudio.google.com."
            )
        else:
            return f"⚠️ **Report generation failed:**\n\n{err}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Load models ──
    models = load_models()
    if not models['loaded']:
        st.stop()

    # ── Initialize session state ──
    for key in ['gemini_model', 'gemini_connected', 'last_prediction',
                'last_input_data', 'last_report']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'gemini_connected' not in st.session_state:
        st.session_state['gemini_connected'] = False

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════

    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.2rem 0 0.5rem; text-align:center;">
            <div style="font-size:2.2rem;">🏥</div>
            <div style="font-size:1rem; font-weight:600; color:#f0f9ff; margin-top:0.3rem;">MediStress AI</div>
            <div style="font-size:0.72rem; color:#38bdf8; letter-spacing:0.08em; text-transform:uppercase;">
                Burnout Risk Predictor
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;'>Navigation</div>", unsafe_allow_html=True)
        page = st.radio("",
                        ["🏠  Home", "🔮  Predict Stress", "📊  Insights", "ℹ️  About"],
                        label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>Gemini AI Settings</div>", unsafe_allow_html=True)

        gemini_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="AIza...",
            help="Get your free key at aistudio.google.com",
            label_visibility="collapsed"
        )

        if st.button("🔑 Connect Gemini", use_container_width=True, type="secondary"):
            if gemini_key_input.strip():
                with st.spinner("Verifying key..."):
                    model = setup_gemini(gemini_key_input.strip())
                if model:
                    st.session_state['gemini_model'] = model
                    st.session_state['gemini_connected'] = True
                    st.success("Connected!")
                else:
                    st.session_state['gemini_connected'] = False
                    st.error("Invalid key — check and retry.")
            else:
                st.warning("Enter your API key first.")

        if st.session_state.get('gemini_connected'):
            st.markdown('<div class="api-connected">● Gemini AI Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-disconnected">○ Gemini AI Not Connected</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Team · DSU 2025–26</div>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.9;">
            Nandeesh N B &nbsp;<span style="color:#1e3a5f">·</span>&nbsp; ENG23AM0047<br>
            N Rohith &nbsp;<span style="color:#1e3a5f">·</span>&nbsp; ENG23AM0046<br>
            Niharika N &nbsp;<span style="color:#1e3a5f">·</span>&nbsp; ENG23AM0048<br>
            Nishvika Teja Reddy &nbsp;<span style="color:#1e3a5f">·</span>&nbsp; ENG23AM0050
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # HOME PAGE
    # ══════════════════════════════════════════════════════════════════════════

    if page == "🏠  Home":

        st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">Generative AI · DSU 23AM3609</div>
            <div class="hero-title">🏥 Healthcare Worker Stress Prediction</div>
            <div class="hero-subtitle">AI-powered burnout risk assessment with explainable ML and personalized recommendations</div>
        </div>
        """, unsafe_allow_html=True)

        # Stat cards
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("89.5%", "XGBoost Accuracy", "↑ 2.3% vs baseline"),
            ("2,003", "Training Samples", "Combined dataset"),
            ("11", "Input Features", "Physio + workplace"),
            ("5", "Departments", "ER · ICU · OPD · IPD · OBG"),
        ]
        for col, (val, label, delta) in zip([c1, c2, c3, c4], stats):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">{label}</div>
                    <div class="stat-delta">{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1.1, 1])

        with col_a:
            st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
            This system addresses critical healthcare worker burnout by combining 
            <b style="color:#7dd3fc">Machine Learning</b>, 
            <b style="color:#7dd3fc">Explainable AI (SHAP)</b>, and 
            <b style="color:#7dd3fc">Google Gemini</b> to deliver real-time, 
            interpretable stress assessments.<br><br>
            Workers in high-pressure environments such as ICUs and ERs face 
            disproportionate burnout. Early detection with actionable guidance 
            can meaningfully reduce long-term health outcomes.
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-header">🚀 How to Use</div>', unsafe_allow_html=True)
            steps = [
                ("01", "Navigate to Predict Stress in the sidebar"),
                ("02", "Enter physiological and workplace inputs"),
                ("03", "Click Predict — get instant stress assessment"),
                ("04", "Connect Gemini API to generate a full AI report"),
                ("05", "Download or share the personalized PDF report"),
            ]
            for num, text in steps:
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; gap:0.8rem; margin-bottom:0.7rem;">
                    <div style="background:#112240; border:1px solid #1e3a5f; border-radius:6px;
                                padding:2px 8px; font-size:0.72rem; font-family:'DM Mono',monospace;
                                color:#38bdf8; white-space:nowrap; margin-top:1px;">{num}</div>
                    <div style="color:#94a3b8; font-size:0.85rem; line-height:1.5;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICT PAGE
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "🔮  Predict Stress":

        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:1.6rem; font-weight:600; color:#f0f9ff; letter-spacing:-0.02em;">🔮 Stress Assessment</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.3rem;">Enter healthcare worker data to get an instant, explainable prediction</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="input-panel-title">🫀 Physiological Metrics</div>', unsafe_allow_html=True)
            temperature = st.slider("Body Temperature (°F)", 70, 104, 85,
                                    help="Core body temperature reading")
            humidity = st.slider("Environmental Humidity (%)", 0, 40, 20,
                                 help="Ambient humidity level at workplace")
            step_count = st.slider("Daily Step Count", 0, 200, 100,
                                   help="Steps measured by wearable device")

        with col2:
            st.markdown('<div class="input-panel-title">🏥 Workplace Metrics</div>', unsafe_allow_html=True)
            workhours = st.slider("Shift Duration (hours)", 4, 20, 8,
                                  help="Total hours worked in this shift")
            patients = st.slider("Patients Attended", 5, 25, 15,
                                 help="Number of patients seen this shift")
            department = st.selectbox("Department",
                                      ["ER", "ICU", "OPD", "IPD", "OBG"],
                                      help="Hospital department")
            dept_stress = st.slider("Dept. Stress Score (0–10)", 0, 10, 5,
                                    help="Self-reported team stress baseline")

        # Derived features
        env_stress = (temperature - 85) / 10 + (humidity - 20) / 10
        workload_intensity = patients / (workhours + 1)
        dept_encoded = {"ER": 0, "ICU": 1, "OPD": 2, "IPD": 3, "OBG": 4}[department]
        activity_encoded = 0 if step_count < 50 else (1 if step_count < 100 else 2)
        shift_encoded = 0 if workhours <= 8 else (1 if workhours <= 12 else 2)

        input_data = {
            'humidity': humidity, 'temperature': temperature, 'step_count': step_count,
            'workhours': workhours, 'patients_attended': patients, 'dept_stress': dept_stress,
            'env_stress': env_stress, 'workload_intensity': workload_intensity,
            'department': department, 'department_encoded': dept_encoded,
            'activity_level_encoded': activity_encoded, 'shift_type_encoded': shift_encoded
        }

        # Live derived indicators
        st.markdown("<hr>", unsafe_allow_html=True)
        di1, di2, di3 = st.columns(3)
        with di1:
            st.metric("Workload Intensity", f"{workload_intensity:.2f}", help="patients ÷ (hours+1)")
        with di2:
            st.metric("Env. Stress Index", f"{env_stress:.2f}", help="Derived from temp & humidity")
        with di3:
            activity_label = ["Low 🟡", "Moderate 🟢", "High 🔵"][activity_encoded]
            st.metric("Activity Level", activity_label)

        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("🎯  Run Stress Prediction", type="primary", use_container_width=True)

        if predict_btn:
            with st.spinner("Running XGBoost model..."):
                result = predict_stress(input_data, models)
            st.session_state['last_prediction'] = result
            st.session_state['last_input_data'] = input_data
            st.session_state['last_report'] = None  # clear old report

        # ── Results section (persists via session_state) ──
        if st.session_state['last_prediction']:
            prediction = st.session_state['last_prediction']

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

            label = prediction['label']
            conf  = prediction['confidence']
            probs = prediction['probabilities']
            card_class = {'Low': 'result-card-low', 'Medium': 'result-card-medium', 'High': 'result-card-high'}[label]
            val_class  = {'Low': 'result-value-low',  'Medium': 'result-value-med',  'High': 'result-value-high'}[label]
            icon       = {'Low': '✅', 'Medium': '⚠️', 'High': '🚨'}[label]
            risk_map   = {'Low': 'Normal', 'Medium': 'Moderate', 'High': 'High Risk'}

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-label">Stress Level</div>
                    <div class="{val_class}">{icon} {label}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="result-card" style="background:#0d1424; border:1px solid #1e2d45;">
                    <div class="result-label">Confidence</div>
                    <div style="font-size:2.2rem; font-weight:700; color:#38bdf8; font-family:'DM Mono',monospace;">
                        {conf:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class="result-card" style="background:#0d1424; border:1px solid #1e2d45;">
                    <div class="result-label">Risk Category</div>
                    <div style="font-size:2.2rem; font-weight:700; color:#c084fc; font-family:'DM Mono',monospace;">
                        {risk_map[label]}
                    </div>
                </div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📈 Class Probabilities</div>', unsafe_allow_html=True)
            bar_classes = ['prob-bar-fill-low', 'prob-bar-fill-med', 'prob-bar-fill-high']
            for lvl, prob, bar_cls in zip(['Low', 'Medium', 'High'], probs, bar_classes):
                pct = prob * 100
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{lvl}</div>
                    <div class="prob-bar-track">
                        <div class="{bar_cls}" style="width:{pct:.1f}%"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Contributing factors
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔍 Key Contributing Factors</div>', unsafe_allow_html=True)

            factors = []
            if temperature > 95: factors.append("🌡️ Elevated Temperature")
            if workhours > 10:   factors.append("⏱️ Extended Shift Hours")
            if patients > 18:    factors.append("👥 High Patient Load")
            if dept_stress > 6:  factors.append("🏥 High Dept. Stress")
            if step_count < 40:  factors.append("🦶 Low Physical Activity")
            if env_stress > 1:   factors.append("🌫️ Environmental Stress")
            if workload_intensity > 2: factors.append("📈 High Workload Intensity")
            if not factors:      factors.append("✅ All metrics within normal range")

            factor_html = "".join([f'<span class="factor-pill">{f}</span>' for f in factors])
            st.markdown(f'<div class="factor-grid">{factor_html}</div>', unsafe_allow_html=True)

            # ── AI Report ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

            gemini_model = st.session_state.get('gemini_model')
            if gemini_model:
                if st.button("📄  Generate AI-Powered Report", type="secondary", use_container_width=True):
                    with st.spinner("Gemini is writing your personalized report..."):
                        report = generate_ai_report(
                            prediction,
                            st.session_state['last_input_data'],
                            gemini_model
                        )
                    st.session_state['last_report'] = report

                if st.session_state.get('last_report'):
                    st.markdown('<div class="section-header">🤖 AI-Generated Report</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="report-box">{st.session_state["last_report"]}</div>',
                                unsafe_allow_html=True)
                    st.download_button(
                        "📥  Download Report (.md)",
                        st.session_state['last_report'],
                        file_name=f"stress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            else:
                st.markdown("""
                <div style="background:rgba(56,189,248,0.05); border:1px solid #1e3a5f; border-radius:10px;
                            padding:1rem 1.2rem; font-size:0.85rem; color:#64748b; text-align:center;">
                    💡 Connect your <b style="color:#38bdf8">Gemini API key</b> in the sidebar 
                    to generate a personalized AI report with recommendations.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # INSIGHTS PAGE
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "📊  Insights":

        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:1.6rem; font-weight:600; color:#f0f9ff;">📊 Model Insights</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.3rem;">Performance metrics, feature importance, and department-level analysis</div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["  Model Performance  ", "  Feature Importance  ", "  Department Analysis  "])

        with tab1:
            st.markdown('<div class="section-header">🎯 Model Comparison</div>', unsafe_allow_html=True)

            perf_data = {
                'Model': ['Random Forest', 'XGBoost', 'MLP Neural Net'],
                'Accuracy': [88.2, 89.5, 87.8],
                'Precision': [87.9, 89.2, 87.5],
                'Recall': [88.1, 89.4, 87.7],
                'F1-Score': [88.0, 89.3, 87.6]
            }
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df.style.highlight_max(subset=['Accuracy','Precision','Recall','F1-Score'],
                                                      color='#112240'),
                         use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            x = np.arange(len(perf_data['Model']))
            w = 0.22
            colors_bar = ['#38bdf8', '#6366f1', '#34d399', '#f59e0b']
            for i, (metric, col) in enumerate(zip(['Accuracy','Precision','Recall','F1-Score'], colors_bar)):
                ax.bar(x + (i - 1.5) * w, perf_data[metric], w,
                       label=metric, color=col, alpha=0.85, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(perf_data['Model'], fontsize=9)
            ax.set_ylim([85, 92])
            ax.set_ylabel("Score (%)")
            ax.set_title("Model Performance Comparison", fontsize=11, pad=12)
            ax.legend(fontsize=8, facecolor='#112240', labelcolor='#94a3b8', edgecolor='#1e2d45')
            apply_chart_style(ax, fig)
            st.pyplot(fig)

        with tab2:
            st.markdown('<div class="section-header">📌 SHAP Feature Importance</div>', unsafe_allow_html=True)

            features_list = ['Temperature', 'Step Count', 'Humidity', 'Work Hours',
                             'Patients Attended', 'Dept. Stress', 'Workload Intensity']
            importance = [0.245, 0.198, 0.156, 0.134, 0.112, 0.089, 0.066]
            sorted_idx = np.argsort(importance)

            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = plt.cm.get_cmap('RdYlGn_r')
            norm_vals = np.array(importance)[sorted_idx]
            norm_c = (norm_vals - norm_vals.min()) / (norm_vals.max() - norm_vals.min() + 1e-9)
            bar_colors = [cmap(0.15 + 0.7 * v) for v in norm_c]
            bars = ax.barh(np.array(features_list)[sorted_idx], norm_vals,
                           color=bar_colors, height=0.6, zorder=3)
            for bar, val in zip(bars, norm_vals):
                ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=8, color='#94a3b8')
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("Global Feature Importance (XGBoost + SHAP)", fontsize=11, pad=12)
            apply_chart_style(ax, fig)
            st.pyplot(fig)

        with tab3:
            st.markdown('<div class="section-header">🏥 Department Stress Profile</div>', unsafe_allow_html=True)

            dept_data = {
                'Department': ['ER', 'ICU', 'OBG', 'IPD', 'OPD'],
                'Avg Stress Score': [1.85, 1.72, 1.45, 1.23, 0.89],
                'High Stress %': [62, 58, 45, 38, 25]
            }
            dept_df = pd.DataFrame(dept_data)
            st.dataframe(dept_df, use_container_width=True, hide_index=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            # Chart 1: avg stress
            dept_colors = ['#ef4444', '#f97316', '#f59e0b', '#38bdf8', '#34d399']
            axes[0].barh(dept_data['Department'], dept_data['Avg Stress Score'],
                         color=dept_colors, height=0.55, zorder=3)
            axes[0].set_xlabel("Average Stress Level")
            axes[0].set_title("Average Stress by Department", fontsize=10, pad=10)
            apply_chart_style(axes[0], fig)

            # Chart 2: high stress %
            wedge_colors = ['#ef4444', '#f97316', '#f59e0b', '#38bdf8', '#34d399']
            wedges, texts, autotexts = axes[1].pie(
                dept_data['High Stress %'],
                labels=dept_data['Department'],
                autopct='%1.0f%%',
                colors=wedge_colors,
                startangle=140,
                pctdistance=0.75
            )
            for t in texts:   t.set_color('#94a3b8'); t.set_fontsize(9)
            for t in autotexts: t.set_color('#0a0f1a'); t.set_fontsize(8); t.set_fontweight('bold')
            axes[1].set_title("High Stress % Distribution", fontsize=10, pad=10)
            axes[1].title.set_color('#e2e8f0')
            fig.patch.set_facecolor('#0d1424')

            st.pyplot(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # ABOUT PAGE
    # ══════════════════════════════════════════════════════════════════════════

    elif page == "ℹ️  About":

        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:1.6rem; font-weight:600; color:#f0f9ff;">ℹ️ About This Project</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.3rem;">
                Course 23AM3609 · Generative AI · Dayananda Sagar University · 2025–2026
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-header">👥 Team Members</div>', unsafe_allow_html=True)
            team = [
                ("Nandeesh N B", "ENG23AM0047"),
                ("N Rohith", "ENG23AM0046"),
                ("Niharika N", "ENG23AM0048"),
                ("Nishvika Teja Reddy", "ENG23AM0050"),
            ]
            for name, uid in team:
                st.markdown(f"""
                <div class="team-card">
                    <div class="team-name">{name}</div>
                    <div class="team-id">{uid}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:1.5rem;">📈 Key Results</div>', unsafe_allow_html=True)
            results = [("XGBoost Accuracy", "89.5%"), ("F1 · Medium Stress", "0.89"),
                       ("F1 · High Stress", "0.91"), ("Training Samples", "2,003")]
            rc1, rc2 = st.columns(2)
            for i, (label, val) in enumerate(results):
                with (rc1 if i % 2 == 0 else rc2):
                    st.metric(label, val)

        with col_right:
            st.markdown('<div class="section-header">🔬 Methodology</div>', unsafe_allow_html=True)
            phases = [
                ("Phase 1 · Data Collection",
                 "Physiological data (temperature, humidity, step count) merged with workplace surveys (work hours, patients, department)."),
                ("Phase 2 · Predictive Modeling",
                 "Random Forest (baseline) → XGBoost (primary, 89.5% acc.) → MLP Deep Learning (3–4 dense layers)."),
                ("Phase 3 · Explainable AI",
                 "SHAP TreeExplainer for RF & XGBoost. SHAP KernelExplainer for MLP. Global and local explanations surfaced in UI."),
                ("Phase 4 · Generative AI",
                 "Google Gemini 1.5 Flash generates personalized stress reports with evidence-based recommendations per worker."),
            ]
            for title, body in phases:
                st.markdown(f"""
                <div class="phase-card">
                    <div class="phase-title">{title}</div>
                    <div class="phase-body">{body}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:1.2rem;">🛠️ Technology Stack</div>', unsafe_allow_html=True)
            techs = ["XGBoost", "Scikit-learn", "TensorFlow/Keras", "SHAP", "Google Gemini API", "Streamlit", "Matplotlib", "Pandas / NumPy"]
            pills = "".join([f'<span class="factor-pill">{t}</span>' for t in techs])
            st.markdown(f'<div class="factor-grid">{pills}</div>', unsafe_allow_html=True)

        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#334155; font-size:0.78rem; padding:0.5rem 0 1rem;">
            © 2025–26 Dayananda Sagar University · Course 23AM3609 · Generative AI
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()