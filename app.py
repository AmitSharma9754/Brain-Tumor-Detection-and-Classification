"""
app.py
------
Brain Tumor Detection and Classification System
CNN + Grad-CAM based MRI tumor classifier — Streamlit frontend.

Developers: Amit Sharma & Krishna Gupta

Run with:
    streamlit run app.py

Expects a trained model at: model/app.keras
(train it first with train_model.ipynb)
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import preprocess_image, IMG_SIZE
from model_utils import make_gradcam_heatmap, overlay_heatmap, CLASS_NAMES, LAST_CONV_LAYER_NAME

MODEL_PATH = r"C:\Users\amits\Downloads\Cornerstone\model\app.keras"
TRAIN_DIR = r"C:\Users\amits\Downloads\Cornerstone\dataset\Training"
TEST_DIR = r"C:\Users\amits\Downloads\Cornerstone\dataset\Testing"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection & Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# GLOBAL STYLE — Cyber Navy / Electric Cyan / Crimson (same theme as reference)
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
    --cyan: #00d9ff;
    --cyan-light: #9df3ff;
    --deep-navy: #050b1a;
    --navy: #0a1128;
    --electric-blue: #0466c8;
    --crimson: #ff3b5c;
    --emerald: #2ecc71;
    --amber: #ffb020;
    --off-white: #eaf6ff;
}

html, body{
    font-family: 'Inter', sans-serif;
    color: var(--off-white) !important;
}

.stApp{
    background: radial-gradient(circle at top left, #0d1b3d 0%, #050b1a 45%, #000000 100%);
    color: var(--off-white) !important;
}

#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

header[data-testid="stHeader"]{
    background: var(--deep-navy) !important;
    background-color: var(--deep-navy) !important;
}

div[data-testid="stDecoration"]{ background: var(--deep-navy) !important; }
div[data-testid="stToolbar"]{ background: transparent !important; }

div[data-testid="stStatusWidget"]{
    background-color: var(--navy) !important;
    border: 1px solid rgba(0,217,255,0.35) !important;
    border-radius: 8px !important;
    color: var(--off-white) !important;
}
div[data-testid="stStatusWidget"] * { color: var(--off-white) !important; fill: var(--off-white) !important; }
div[data-testid="stStatusWidget"] button{ background-color: var(--electric-blue) !important; border: 1px solid var(--cyan) !important; }

.stMarkdown, .stMarkdown p, .stText, .stCaption, .stDataFrame, .stDataFrame *,
.stSelectbox label, .stSlider label, .stFileUploader label, .stButton label, .stRadio label {
    color: var(--off-white) !important;
}

.hero-banner{
    background: linear-gradient(135deg, var(--navy) 0%, var(--electric-blue) 55%, var(--navy) 100%);
    border: 1px solid var(--cyan);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 22px;
    box-shadow: 0 0 25px rgba(0,217,255,0.25), inset 0 0 40px rgba(4,102,200,0.2);
    text-align: center;
}
.hero-title{
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--cyan-light), var(--cyan), #d0f7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.hero-subtitle{ color: var(--cyan-light) !important; font-size: 1.05rem; opacity: 0.9; }

.cyan-divider{
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    margin: 14px 0 22px 0;
    border: none;
}

.glass-card{
    background: linear-gradient(160deg, #0f1c3d 0%, #060d1e 100%);
    border: 1px solid rgba(0,217,255,0.35);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    color: var(--off-white) !important;
}
.glass-card h3, .glass-card h4, .glass-card p, .glass-card li { color: var(--off-white) !important; }

.metric-card{
    background: linear-gradient(160deg, #0f1c3d, #050b1a);
    border: 1px solid var(--cyan);
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 15px rgba(4,102,200,0.3);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.metric-value{ font-size: 2rem; font-weight: 700; color: var(--cyan); font-family: 'Orbitron', sans-serif; margin: 0; }
.metric-label{ color: var(--cyan-light); opacity: 0.9; font-size: 0.85rem; letter-spacing: 0.5px; text-transform: uppercase; margin: 0; }

.result-card{
    background: linear-gradient(135deg, #0d2b1a, #0d1c33);
    border: 2px solid var(--cyan);
    border-radius: 16px;
    padding: 26px;
    text-align: center;
    box-shadow: 0 0 25px rgba(0,217,255,0.35);
}
.result-card.danger{ border-color: var(--crimson); box-shadow: 0 0 25px rgba(255,59,92,0.35); }
.result-emoji{ font-size: 3rem; }
.result-text{ font-family:'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700; margin-top:6px; color: var(--cyan); }
.result-card.danger .result-text{ color: var(--crimson); }
.result-sub{ color: var(--cyan-light); opacity:0.85; margin-top:4px; }

.stButton>button{
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    color: #041022;
    border: 1px solid var(--cyan-light);
    border-radius: 10px;
    padding: 10px 26px;
    font-weight: 700;
}
.stButton>button:hover{
    background: linear-gradient(135deg, var(--cyan), var(--cyan-light));
    color: #000814;
    border: 1px solid #fff;
    box-shadow: 0 0 18px rgba(0,217,255,0.6);
}

[data-testid="stFileUploaderDropzone"]{
    background: linear-gradient(160deg, #0f1c3d, #060d1e) !important;
    border: 2px dashed rgba(0,217,255,0.5) !important;
    border-radius: 14px !important;
}

.stTabs [data-baseweb="tab-list"]{
    gap: 6px;
    background: #0a1128;
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(0,217,255,0.35);
}
.stTabs [data-baseweb="tab"]{
    height: 50px;
    background: transparent;
    color: var(--cyan-light) !important;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0 18px;
}
.stTabs [aria-selected="true"]{
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    color: #04101f !important;
    box-shadow: 0 0 12px rgba(0,217,255,0.45);
    border: 1px solid var(--cyan-light);
}

.footer-sig{
    text-align:center;
    padding: 14px;
    margin-top: 30px;
    color: var(--cyan-light);
    opacity: 0.9;
    font-size: 0.85rem;
    letter-spacing: 1px;
    border-top: 1px solid rgba(0,217,255,0.25);
}

.dev-card{
    background: linear-gradient(160deg, #0f1c3d, #050b1a);
    border: 1px solid var(--cyan);
    border-radius: 18px;
    padding: 26px;
    text-align:center;
    box-shadow: 0 0 20px rgba(4,102,200,0.3);
}
.dev-avatar{
    width: 74px; height: 74px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    display:flex; align-items:center; justify-content:center;
    font-size: 1.8rem; font-weight:800; color:#04101f;
    margin: 0 auto 12px auto;
    font-family:'Orbitron', sans-serif;
}
.dev-name{ font-family:'Orbitron', sans-serif; font-size: 1.3rem; font-weight:700; color: var(--cyan-light); margin-bottom: 2px; }
.dev-role{ opacity:0.8; font-size:0.85rem; margin-bottom: 14px; }

.step-item{ display:flex; align-items:flex-start; gap:14px; padding: 12px 4px; border-bottom: 1px solid rgba(0,217,255,0.15); }
.step-item:last-child{ border-bottom:none; }
.step-num{
    flex-shrink:0; width:32px; height:32px; border-radius:50%;
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    border:1px solid var(--cyan-light);
    display:flex; align-items:center; justify-content:center;
    font-weight:700; color:#04101f; font-size:0.9rem;
}
.step-text{ padding-top:5px; font-size:0.95rem; }

.feature-grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-top: 10px; }
.feature-tile{
    background: linear-gradient(160deg,#0f1c3d,#050b1a);
    border:1px solid rgba(0,217,255,0.35);
    border-radius:14px; padding:18px; text-align:center;
}
.feature-icon{ font-size:1.8rem; margin-bottom:6px; }
.feature-title{ font-weight:700; color:var(--cyan-light); margin-bottom:4px; font-size:0.9rem; }
.feature-desc{ font-size:0.78rem; opacity:0.8; line-height:1.4; }

.disclaimer-box{
    background: linear-gradient(135deg, #2b1c0d, #1a0810);
    border: 1px solid var(--amber);
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 0.9rem;
    box-shadow: 0 0 15px rgba(255,176,32,0.25);
}

.tech-pill{
    display:inline-block;
    background: linear-gradient(135deg, #0f1c3d, #050b1a);
    border: 1px solid var(--cyan);
    color: var(--cyan-light);
    border-radius: 999px;
    padding: 6px 16px;
    margin: 4px 6px 4px 0;
    font-size: 0.82rem;
    font-weight: 600;
}

.timeline-item{
    display:flex; gap:14px; padding: 10px 4px;
}
.timeline-dot{
    flex-shrink:0; width:14px; height:14px; border-radius:50%;
    background: var(--cyan); margin-top:5px;
    box-shadow: 0 0 10px rgba(0,217,255,0.7);
}
.timeline-text{ font-size:0.92rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HERO
# ============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🧠 Brain Tumor Detection and Classification System</div>
    <div class="hero-subtitle">AI-Powered MRI Analysis using a Convolutional Neural Network with Grad-CAM Explainability</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    _ = model(dummy_input)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_load_error = e

if not model_loaded:
    st.error(
        f"❌ Could not load model from `{MODEL_PATH}`.\n\n"
        f"Train it first by running the `train_model.ipynb` notebook."
    )
    st.exception(model_load_error)
    st.stop()

CLASS_INFO = {
    "glioma": ("🔴", "Glioma Tumor", "A tumor that arises from glial cells in the brain or spine."),
    "meningioma": ("🟠", "Meningioma Tumor", "A tumor forming in the meninges, the membranes covering the brain and spinal cord."),
    "notumor": ("🟢", "No Tumor", "No tumor detected — the scan appears normal."),
    "pituitary": ("🟣", "Pituitary Tumor", "A tumor that forms in the pituitary gland at the base of the brain."),
}
CLASS_COLORS = {"glioma": "#ff3b5c", "meningioma": "#ffb020", "notumor": "#2ecc71", "pituitary": "#9d6bff"}


@st.cache_data(show_spinner=False)
def count_images_per_class(base_dir):
    """Counts images per class folder. Returns dict or None if dir missing."""
    if not os.path.isdir(base_dir):
        return None
    counts = {}
    for cname in CLASS_NAMES:
        folder = os.path.join(base_dir, cname)
        if os.path.isdir(folder):
            n = len([f for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            counts[cname] = n
        else:
            counts[cname] = 0
    return counts


@st.cache_data(show_spinner=False)
def get_sample_image_paths(base_dir, per_class=1):
    """Grabs sample image paths per class for a preview grid."""
    samples = {}
    if not os.path.isdir(base_dir):
        return samples
    for cname in CLASS_NAMES:
        folder = os.path.join(base_dir, cname)
        files = glob.glob(os.path.join(folder, "*.jpg")) + \
                glob.glob(os.path.join(folder, "*.jpeg")) + \
                glob.glob(os.path.join(folder, "*.png"))
        if files:
            samples[cname] = files[:per_class]
    return samples


@st.cache_data(show_spinner=False)
def get_mean_image_per_class(base_dir, n_samples=25):
    """
    Computes the pixel-wise average image for each class (a quick visual
    fingerprint of what each tumor class 'looks like on average').
    """
    from PIL import Image
    means = {}
    if not os.path.isdir(base_dir):
        return means
    for cname in CLASS_NAMES:
        folder = os.path.join(base_dir, cname)
        files = (glob.glob(os.path.join(folder, "*.jpg")) +
                 glob.glob(os.path.join(folder, "*.jpeg")) +
                 glob.glob(os.path.join(folder, "*.png")))[:n_samples]
        if not files:
            continue
        acc = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)
        count = 0
        for f in files:
            try:
                img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                acc += np.array(img, dtype=np.float64)
                count += 1
            except Exception:
                continue
        if count > 0:
            means[cname] = (acc / count).astype(np.uint8)
    return means


def run_test_set_evaluation(_model, test_dir, max_per_class=30, batch_size=32):
    """
    Evaluates the model on the Testing folder using BATCHED prediction
    (much faster than predicting one image at a time). Capped per class
    to keep this quick.
    """
    all_batches = []
    all_labels = []

    for idx, cname in enumerate(CLASS_NAMES):
        folder = os.path.join(test_dir, cname)
        if not os.path.isdir(folder):
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                        glob.glob(os.path.join(folder, "*.jpeg")) +
                        glob.glob(os.path.join(folder, "*.png")))[:max_per_class]
        for fpath in files:
            img_batch, _ = preprocess_image(fpath)
            all_batches.append(img_batch[0])
            all_labels.append(idx)

    if not all_batches:
        return [], []

    X = np.stack(all_batches, axis=0)
    preds = _model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = list(np.argmax(preds, axis=1))
    return all_labels, y_pred


# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predict",
    "📊 Data Visualization",
    "🎯 Accuracy",
    "ℹ️ About",
])

# ----------------------------------------------------------------------------
# TAB 1 — PREDICT
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
            📤 Upload a Brain MRI Scan
        </h4>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an MRI image (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Analyzing MRI scan with the CNN model..."):
            img_batch, original_img = preprocess_image(uploaded_file)

            preds = model.predict(img_batch, verbose=0)
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            predicted_class = CLASS_NAMES[idx]

            heatmap = make_gradcam_heatmap(img_batch, model, LAST_CONV_LAYER_NAME, pred_index=idx)
            overlay = overlay_heatmap(original_img, heatmap)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:4px 0; font-weight:bold;">
                    🩻 Uploaded MRI
                </h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(original_img, width="stretch")

        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:4px 0; font-weight:bold;">
                    🔥 Grad-CAM Heatmap
                </h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(overlay, width="stretch", caption="Regions the model focused on")

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        emoji, label, desc = CLASS_INFO.get(predicted_class, ("❓", predicted_class, ""))
        card_class = "result-card" if predicted_class == "notumor" else "result-card danger"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-text">{label}</div>
            <div class="result-sub">Confidence: {confidence * 100:.2f}%</div>
            <div class="result-sub" style="margin-top:8px; opacity:0.75;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        gc1, gc2, gc3, gc4 = st.columns(4)
        for gc, cname in zip([gc1, gc2, gc3, gc4], CLASS_NAMES):
            prob = float(preds[0][CLASS_NAMES.index(cname)])
            with gc:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.3rem;">{prob*100:.1f}%</div>
                    <div class="metric-label">{cname}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                📈 Class Probability Distribution
            </h4>
        </div>
        """, unsafe_allow_html=True)

        prob_fig = px.bar(
            x=CLASS_NAMES, y=[float(p) for p in preds[0]],
            labels={"x": "Class", "y": "Probability"},
            color=CLASS_NAMES,
            color_discrete_map=CLASS_COLORS,
        )
        prob_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff", showlegend=False,
        )
        st.plotly_chart(prob_fig, width="stretch")

        with st.expander("🔍 Raw prediction values"):
            st.json({CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))})
    else:
        st.info("👆 Upload an MRI image above to get started.")

# ----------------------------------------------------------------------------
# TAB 2 — DATA VISUALIZATION
# ----------------------------------------------------------------------------
with tab2:
    train_counts = count_images_per_class(TRAIN_DIR)
    test_counts = count_images_per_class(TEST_DIR)

    if train_counts is None:
        st.error(f"❌ Could not find Training folder at `{TRAIN_DIR}`. "
                 f"Make sure the dataset is in place to see visualizations.")
    else:
        # ---- Row 1: Pie + Grouped bar ----
        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                🥧 Class Distribution — Training Set
            </h4>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            fig_pie = px.pie(
                names=list(train_counts.keys()),
                values=list(train_counts.values()),
                color=list(train_counts.keys()),
                color_discrete_map=CLASS_COLORS,
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#eaf6ff",
                legend=dict(font=dict(color="#eaf6ff")),
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_color="#050b1a")
            st.plotly_chart(fig_pie, width="stretch")

        with colB:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    📊 Training vs Testing Counts
                </h4>
            </div>
            """, unsafe_allow_html=True)
            count_df = pd.DataFrame({
                "Class": CLASS_NAMES * 2,
                "Count": [train_counts[c] for c in CLASS_NAMES] + [test_counts[c] for c in CLASS_NAMES],
                "Split": ["Training"] * 4 + ["Testing"] * 4,
            })
            fig_bar = px.bar(
                count_df, x="Class", y="Count", color="Split", barmode="group",
                color_discrete_map={"Training": "#00d9ff", "Testing": "#ff3b5c"},
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eaf6ff",
                legend=dict(font=dict(color="#eaf6ff")),
            )
            st.plotly_chart(fig_bar, width="stretch")

        # ---- Metric cards ----
        total_train = sum(train_counts.values())
        total_test = sum(test_counts.values()) if test_counts else 0
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_train}</div>
            <div class="metric-label">Training Images</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_test}</div>
            <div class="metric-label">Testing Images</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_train + total_test}</div>
            <div class="metric-label">Total Images</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">4</div>
            <div class="metric-label">Classes</div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        # ---- Row 2: Treemap + Horizontal % bar ----
        colC, colD = st.columns(2)
        with colC:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    🌳 Tumor vs No-Tumor Breakdown
                </h4>
            </div>
            """, unsafe_allow_html=True)
            tree_df = pd.DataFrame({
                "Category": ["Tumor" if c != "notumor" else "No Tumor" for c in CLASS_NAMES],
                "Class": CLASS_NAMES,
                "Count": [train_counts[c] + test_counts.get(c, 0) for c in CLASS_NAMES],
            })
            fig_tree = px.treemap(
                tree_df, path=["Category", "Class"], values="Count",
                color="Class", color_discrete_map=CLASS_COLORS,
            )
            fig_tree.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#050b1a",
                margin=dict(t=10, l=10, r=10, b=10),
            )
            st.plotly_chart(fig_tree, width="stretch")

        with colD:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    📐 Class Share (% of Total)
                </h4>
            </div>
            """, unsafe_allow_html=True)
            total_all = total_train + total_test
            pct_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Percent": [round((train_counts[c] + test_counts.get(c, 0)) / total_all * 100, 1) for c in CLASS_NAMES],
            }).sort_values("Percent")
            fig_hbar = px.bar(
                pct_df, x="Percent", y="Class", orientation="h",
                color="Class", color_discrete_map=CLASS_COLORS,
                text="Percent",
            )
            fig_hbar.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_hbar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff", showlegend=False,
            )
            st.plotly_chart(fig_hbar, width="stretch")

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        # ---- Row 3: Mean/average image per class ----
        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                🧬 Average MRI Signature per Class
            </h4>
            <p style="text-align:center; opacity:0.8; margin:0; font-size:0.85rem;">
                Pixel-wise average of ~25 training images per class — a visual fingerprint of each tumor type.
            </p>
        </div>
        """, unsafe_allow_html=True)
        mean_imgs = get_mean_image_per_class(TRAIN_DIR, n_samples=25)
        if mean_imgs:
            mcols = st.columns(4)
            for mcol, cname in zip(mcols, CLASS_NAMES):
                with mcol:
                    if cname in mean_imgs:
                        st.image(mean_imgs[cname], caption=f"Avg — {cname}", width="stretch")

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        # ---- Row 4: Sample MRI grid (4 per class) ----
        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                🖼️ Sample MRI Scans per Class
            </h4>
        </div>
        """, unsafe_allow_html=True)

        samples = get_sample_image_paths(TRAIN_DIR, per_class=4)
        for cname in CLASS_NAMES:
            if cname in samples and samples[cname]:
                st.caption(f"**{cname}**")
                scols = st.columns(4)
                for scol, fpath in zip(scols, samples[cname]):
                    with scol:
                        st.image(fpath, width="stretch")

# ----------------------------------------------------------------------------
# TAB 3 — ACCURACY
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("""
    <div class="glass-card">
        <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
            🎯 Model Accuracy on Test Set
        </h4>
        <p style="text-align:center; opacity:0.85; margin:0;">
            Runs the CNN on a sample of images from the Testing folder (batched prediction — fast)
            and compares predictions against the true labels.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample_size = st.slider(
        "Images per class to evaluate (lower = faster)",
        min_value=10, max_value=100, value=25, step=5,
    )
    run_eval = st.button("▶️ Run Evaluation on Test Set", width="stretch")

    if run_eval:
        if not os.path.isdir(TEST_DIR):
            st.error(f"❌ Could not find Testing folder at `{TEST_DIR}`.")
        else:
            with st.spinner(f"Evaluating model on ~{sample_size * len(CLASS_NAMES)} test images (batched)..."):
                y_true, y_pred = run_test_set_evaluation(model, TEST_DIR, max_per_class=sample_size)
                st.session_state["eval_y_true"] = y_true
                st.session_state["eval_y_pred"] = y_pred

    if "eval_y_true" in st.session_state:
        y_true = st.session_state["eval_y_true"]
        y_pred = st.session_state["eval_y_pred"]

        overall_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{overall_acc*100:.2f}%</div>
            <div class="metric-label">Overall Accuracy</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{report['weighted avg']['precision']*100:.1f}%</div>
            <div class="metric-label">Precision</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{report['weighted avg']['recall']*100:.1f}%</div>
            <div class="metric-label">Recall</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{report['weighted avg']['f1-score']*100:.1f}%</div>
            <div class="metric-label">F1 Score</div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    🧊 Confusion Matrix
                </h4>
            </div>
            """, unsafe_allow_html=True)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=CLASS_NAMES, y=CLASS_NAMES,
                color_continuous_scale=["#050b1a", "#0466c8", "#00d9ff"],
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#eaf6ff",
            )
            st.plotly_chart(fig_cm, width="stretch")

        with colB:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    📋 Per-Class Metrics
                </h4>
            </div>
            """, unsafe_allow_html=True)
            report_df = pd.DataFrame(report).transpose().loc[CLASS_NAMES]
            report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
            st.dataframe(report_df, width="stretch")

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                📈 Per-Class F1 Score
            </h4>
        </div>
        """, unsafe_allow_html=True)
        f1_fig = px.bar(
            x=CLASS_NAMES, y=[report[c]["f1-score"] for c in CLASS_NAMES],
            color=CLASS_NAMES, color_discrete_map=CLASS_COLORS,
            labels={"x": "Class", "y": "F1 Score"},
        )
        f1_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff", showlegend=False,
        )
        st.plotly_chart(f1_fig, width="stretch")
    else:
        st.info("👆 Pick a sample size and click **Run Evaluation on Test Set** to compute accuracy, confusion matrix, and per-class metrics.")

# ----------------------------------------------------------------------------
# TAB 4 — ABOUT
# ----------------------------------------------------------------------------
with tab4:
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("""
        <div class="dev-card">
            <div class="dev-avatar">AS</div>
            <div class="dev-name">Amit Sharma</div>
            <div class="dev-role">ML Engineer & Developer</div>
            <p style="font-size:0.85rem; opacity:0.85; margin:0;">
                Brain Tumor Detection and Classification System
            </p>
        </div>
        """, unsafe_allow_html=True)

    with dcol2:
        st.markdown("""
        <div class="dev-card">
            <div class="dev-avatar">KG</div>
            <div class="dev-name">Krishna Gupta</div>
            <div class="dev-role">ML Engineer & Developer</div>
            <p style="font-size:0.85rem; opacity:0.85; margin:0;">
                Brain Tumor Detection and Classification System
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Disclaimer</strong><br>
            This model is trained on a publicly available MRI dataset and reflects
            statistical patterns learned from that data. Predictions are for
            educational and demonstration purposes only and must not be used as
            a substitute for professional medical diagnosis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 10px 0;">🛠️ Technology Stack</h4>
            <div style="text-align:center;">
                <span class="tech-pill">Python</span>
                <span class="tech-pill">TensorFlow / Keras</span>
                <span class="tech-pill">Streamlit</span>
                <span class="tech-pill">OpenCV</span>
                <span class="tech-pill">Plotly</span>
                <span class="tech-pill">NumPy</span>
                <span class="tech-pill">Pandas</span>
                <span class="tech-pill">Scikit-learn</span>
                <span class="tech-pill">Grad-CAM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">🚧 Limitations</h4>
            <p style="margin:2px 0; font-size:0.88rem;">• Trained on a single public dataset — may not generalize to all scanners/protocols.</p>
            <p style="margin:2px 0; font-size:0.88rem;">• Only classifies 4 categories; cannot detect other conditions.</p>
            <p style="margin:2px 0; font-size:0.88rem;">• Not validated for clinical use.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">🔭 Future Scope</h4>
            <p style="margin:2px 0; font-size:0.88rem;">• Tumor size & location segmentation.</p>
            <p style="margin:2px 0; font-size:0.88rem;">• Multi-modal MRI support (T1, T2, FLAIR).</p>
            <p style="margin:2px 0; font-size:0.88rem;">• Deployment as a hospital-integrated web service.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 8px 0;">ℹ️ About the App</h4>
            <p style="text-align:center; margin:0; opacity:0.9;">
                This system classifies <strong>brain MRI scans</strong> into four categories —
                glioma, meningioma, pituitary tumor, or no tumor — using a
                <strong>Convolutional Neural Network</strong> trained with TensorFlow/Keras,
                with <strong>Grad-CAM</strong> visualizations to explain each prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">🧭 How to Use</h4>
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text">Go to the <strong>🔍 Predict</strong> tab.</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text">Upload a brain MRI scan (JPG or PNG).</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text">View the predicted tumor class, confidence score, and Grad-CAM heatmap.</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-text">Explore the <strong>📊 Data Visualization</strong> and <strong>🎯 Accuracy</strong> tabs for dataset and model insights.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">🧩 Project Workflow</h4>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-text">Collect & organize MRI dataset into 4 labeled classes.</div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-text">Preprocess images — resize to 150×150, normalize, augment.</div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-text">Train a custom CNN from scratch using TensorFlow/Keras.</div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-text">Evaluate with accuracy, confusion matrix, and per-class metrics.</div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-text">Add Grad-CAM for visual explainability of predictions.</div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-text">Wrap everything in an interactive Streamlit web app.</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="text-align:center; margin:0 0 10px 0;">✨ What's Inside</h4>
        <div class="feature-grid">
            <div class="feature-tile">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Custom CNN</div>
                <div class="feature-desc">Deep convolutional network trained from scratch on MRI scans</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">🔥</div>
                <div class="feature-title">Grad-CAM</div>
                <div class="feature-desc">Visual explanation of which brain regions influenced the prediction</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Interactive Charts</div>
                <div class="feature-desc">Class distribution, treemap, confusion matrix, and per-class metrics</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Real-Time Inference</div>
                <div class="feature-desc">Instant predictions the moment an MRI scan is uploaded</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="text-align:center; margin:0 0 8px 0;">📚 Dataset & Acknowledgment</h4>
        <p style="text-align:center; margin:0; opacity:0.85; font-size:0.9rem;">
            Trained on a publicly available Brain Tumor MRI dataset (Kaggle), organized into
            <code>Training/</code> and <code>Testing/</code> folders with 4 classes: glioma,
            meningioma, pituitary, and no-tumor. We thank the original dataset contributors
            for making this data openly available for research and education.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer-sig">
    🧠 Brain Tumor Detection and Classification System &nbsp;|&nbsp; Built by Amit Sharma & Krishna Gupta
    &nbsp;|&nbsp; Powered by CNN (TensorFlow/Keras) + Grad-CAM + Streamlit ✨
</div>
""", unsafe_allow_html=True)