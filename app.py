"""
app.py
Generative vs. Discriminative Model Lab (Streamlit)
Clean White, Black & Crimson Red Theme • High-Accuracy Fine-Tuned PyTorch Models
Compatible with both Local Execution and Streamlit Cloud Deployment.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup dynamic path resolution for local and Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path_dir in [BASE_DIR, os.path.join(BASE_DIR, "generative-discriminative-lab")]:
    if os.path.exists(path_dir) and path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from backend.preprocessing import preprocess_image
from backend.inference import (
    load_or_train_classifier,
    classify_digit,
    get_device,
    DEFAULT_CLASSIFIER_PATH
)
from backend.generation import (
    load_or_train_vae,
    generate_digit_samples,
    DEFAULT_VAE_PATH
)
from models.classifier import load_mnist_data

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


# Page Configuration
st.set_page_config(
    page_title="MNIST Lab • Generative vs. Discriminative",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean White, Black & Crimson Red Theme CSS
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #111827;
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #111827 !important;
    }
    
    /* Header */
    .main-header {
        border-bottom: 2.5px solid #111827;
        padding-bottom: 12px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .main-title span {
        color: #DC2626;
    }
    .main-subtitle {
        color: #6B7280 !important;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    
    /* Cards */
    .card-box {
        background: #FAFAFA;
        border: 1.5px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-red {
        border-top: 4px solid #DC2626;
    }
    .card-black {
        border-top: 4px solid #111827;
    }
    .card-heading {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .card-sub {
        font-size: 0.85rem;
        color: #6B7280 !important;
        margin-bottom: 16px;
    }

    /* Metrics */
    .metric-container {
        display: flex;
        gap: 16px;
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px 18px;
        align-items: center;
        margin-top: 12px;
    }
    .metric-digit {
        font-size: 2.4rem;
        font-weight: 800;
        color: #DC2626 !important;
        line-height: 1;
    }
    .metric-meta {
        display: flex;
        flex-direction: column;
    }
    .metric-conf {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-acc {
        font-size: 0.8rem;
        color: #059669 !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton>button:hover {
        background-color: #B91C1C !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #DC2626 !important;
    }
</style>
""", unsafe_allow_html=True)


# Model Loading with Cache
@st.cache_resource(show_spinner=False)
def get_cached_models():
    cnn_model, test_acc = load_or_train_classifier(model_path=DEFAULT_CLASSIFIER_PATH, epochs=3)
    vae_model = load_or_train_vae(model_path=DEFAULT_VAE_PATH, epochs=4)
    hw_device = str(get_device()).upper()
    return cnn_model, test_acc, vae_model, hw_device


with st.spinner("Initializing fine-tuned PyTorch models..."):
    cnn_model, test_accuracy, vae_model, device_name = get_cached_models()


# Top Header
st.markdown(f"""
<div class="main-header">
    <div>
        <div class="main-title">MNIST <span>Lab</span></div>
        <div class="main-subtitle">Discriminative Classification (CNN) & Generative Synthesis (VAE) • PyTorch</div>
    </div>
    <div style="text-align: right;">
        <span style="background: #F3F4F6; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; border: 1px solid #E5E7EB;">
            Device: {device_name}
        </span>
        <span style="background: #FEF2F2; color: #B91C1C; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; border: 1px solid #FECACA; margin-left: 6px;">
            Test Accuracy: {test_accuracy:.2f}%
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# Main 2-Column Layout
col_disc, col_gen = st.columns([1, 1], gap="large")


# =============================================================
# COLUMN 1: DISCRIMINATIVE MODEL (CNN CLASSIFIER)
# =============================================================
with col_disc:
    st.markdown("""
    <div class="card-box card-red">
        <div class="card-heading">1. Digit Classifier (CNN)</div>
        <div class="card-sub">Draw a digit or upload an image to predict class 0–9.</div>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Select Input Method:",
        ["✏️ Draw Digit", "📁 Upload Image", "🔢 Sample Digit"],
        horizontal=True
    )

    input_image_data = None

    if input_mode == "✏️ Draw Digit":
        if CANVAS_AVAILABLE:
            st.caption("Draw a digit (0–9) using your mouse/touch:")
            import inspect
            canvas_params = {
                "fill_color": "#000000",
                "stroke_width": 20,
                "stroke_color": "#FFFFFF",
                "background_color": "#000000",
                "height": 220,
                "width": 220,
                "drawing_mode": "freedraw",
                "key": "mnist_drawing_canvas"
            }
            if "return_image_data" in inspect.signature(st_canvas).parameters:
                canvas_params["return_image_data"] = True

            canvas_result = st_canvas(**canvas_params)

            if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                if np.max(canvas_result.image_data[:, :, :3]) > 10:
                    input_image_data = canvas_result.image_data
        else:
            st.info("Canvas library not available. Please use the Upload or Sample options.")

    elif input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a handwritten digit (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            input_image_data = Image.open(uploaded_file)
            st.image(input_image_data, caption="Uploaded Image", width=120)

    elif input_mode == "🔢 Sample Digit":
        sample_digit = st.selectbox("Choose MNIST Test Digit:", [f"Digit {i}" for i in range(10)], index=3)
        target = int(sample_digit.split()[-1])
        try:
            _, test_loader = load_mnist_data(batch_size=256)
            for imgs, labels in test_loader:
                matches = (labels == target).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    img_t = imgs[matches[0]]
                    unnorm = (img_t.squeeze().numpy() * 0.3081 + 0.1307) * 255.0
                    input_image_data = Image.fromarray(np.clip(unnorm, 0, 255).astype(np.uint8), mode="L")
                    st.image(input_image_data, caption=f"Authentic MNIST Sample: {target}", width=100)
                    break
        except Exception:
            pass

    # Action Button
    classify_clicked = st.button("🚀 Classify Digit", use_container_width=True)

    if classify_clicked or (input_image_data is not None and input_mode == "🔢 Sample Digit"):
        if input_image_data is None:
            st.warning("⚠️ Please draw, upload, or choose a digit first.")
        else:
            try:
                with st.spinner("Preprocessing and classifying..."):
                    cnn_tensor, _, disp_img = preprocess_image(input_image_data)
                    prediction_res = classify_digit(cnn_model, cnn_tensor)

                st.session_state["disp_img"] = disp_img
                st.session_state["prediction_res"] = prediction_res
            except ValueError as ve:
                st.warning(f"⚠️ {str(ve)}")
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

    # Show Prediction Results
    if "prediction_res" in st.session_state:
        res = st.session_state["prediction_res"]
        disp_img = st.session_state["disp_img"]

        pred_val = res["predicted_digit"]
        conf_pct = res["confidence"] * 100.0

        c_img, c_info = st.columns([1, 2.5])
        with c_img:
            st.image(disp_img, caption="Preprocessed 28×28", width=80)
        with c_info:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-digit">{pred_val}</div>
                <div class="metric-meta">
                    <div class="metric-conf">Confidence: {conf_pct:.1f}%</div>
                    <div class="metric-acc">Held-out Accuracy: {test_accuracy:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("**Class Probabilities (0–9):**")
        prob_df = pd.DataFrame({
            "Probability": res["probabilities_list"]
        }, index=[str(i) for i in range(10)])
        st.bar_chart(prob_df, height=200)


# =============================================================
# COLUMN 2: GENERATIVE MODEL (VAE SYNTHESIZER)
# =============================================================
with col_gen:
    st.markdown("""
    <div class="card-box card-black">
        <div class="card-heading">2. Digit Synthesizer (VAE)</div>
        <div class="card-sub">Sample latent vectors z ~ N(0, I) to generate brand-new digits.</div>
    </div>
    """, unsafe_allow_html=True)

    num_samples = st.slider(
        "Number of Samples to Generate:",
        min_value=1,
        max_value=24,
        value=12,
        step=1
    )

    generate_clicked = st.button("🎲 Generate Digits", use_container_width=True)

    if generate_clicked or "vae_samples" not in st.session_state:
        with st.spinner("Sampling VAE latent space..."):
            samples = generate_digit_samples(vae_model, num_samples=num_samples)
            st.session_state["vae_samples"] = samples

    # Grid Display of Generated Digits
    display_list = st.session_state.get("vae_samples", [])[:num_samples]
    cols_per_row = 4
    for i in range(0, len(display_list), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, img in enumerate(display_list[i : i + cols_per_row]):
            with row_cols[j]:
                st.image(img, caption=f"#{i+j+1}", use_container_width=True)
