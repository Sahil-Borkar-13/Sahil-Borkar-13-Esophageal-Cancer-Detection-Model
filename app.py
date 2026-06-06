import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

st.set_page_config(
    page_title="Esophageal Medical CAD Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        height: 3.2em;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 14px 0 rgba(6, 182, 212, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(59, 130, 246, 0.6);
    }
    .clinical-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        padding: 16px;
        border-radius: 12px;
        border-left: 4px solid #06b6d4;
    }
    h1 { font-weight: 800 !important; color: #ffffff !important; letter-spacing: -0.5px; }
    h2, h3, h4 { color: #38bdf8 !important; font-weight: 700 !important; }
    p, li { color: #cbd5e1 !important; font-size: 15px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='text-align: center; color: #06b6d4;'>🔬 CAD Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 12px;'>Clinical Engine Deployment v1.4</p>", unsafe_allow_html=True)
st.sidebar.write("---")

app_mode = st.sidebar.selectbox("Select Diagnostic Protocol:", [
    "📌 Clinical Overview", 
    "🔬 Module Alpha: Automated Tissue Screening", 
    "🎯 Module Beta: Regional Boundary Localization"
])

if app_mode == "📌 Clinical Overview":
    st.markdown("<h1 style='color: #ffffff;'>Esophageal Cancer Detection App</h1>", unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("""
    <div class="clinical-card">
        <h3>Esophageal Intelligent Diagnostic Suite</h3>
        <p>Welcome to the production deployment portal. This computer-aided diagnostic (CAD) system integrates dual-stage uncoupled deep learning pipelines to provide clinical decision support during gastroenterological evaluations.</p>
        <br>
        <h4>Operational Modules & Technical Protocols:</h4>
        <ul>
            <li><b>Module Alpha — Automated Diagnostic Screening (ADS-v1):</b> Executes high-dimensional feature extraction across global tissue maps. It processes macro-textural patterns and localized mucosal irregularities within the endoscopic frame to compute a structural integrity score, flagging potential sub-surface anomalies with statistical confidence.</li>
            <li><b>Module Beta — High-Resolution Boundary Mapping (HRBM-v1):</b> Triggers a localized spatial scanning grid targeting localized Regions of Interest (ROIs). By evaluating pixel-level semantic distributions, it performs sub-pixel edge-delineation to map and trace the precise morphological boundaries of suspicious malignant lesions.</li>
        </ul>
        <br>
        <p style='font-size: 13px; color: #94a3b8;'>🔒 <i>Security Architecture: Core underlying neural network topologies, hyperparameter configurations, and layer-wise weight distributions are strictly obfuscated for deployment integrity and proprietary asset safety.</i></p>
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "🔬 Module Alpha: Automated Tissue Screening":
    st.markdown("<h1>Module Alpha: Automated Tissue Screening</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #06b6d4;'>Analysis Pipeline: Deep Texture Feature Extraction (ADS-v1)</p>", unsafe_allow_html=True)
    st.write("---")
    
    uploaded_file = st.file_uploader("Drop endoscopy frame here or browse files...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown('<div class="clinical-card"><h4>Input Source Frame</h4></div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown('<div class="clinical-card"><h4>Analytics Engine</h4></div>', unsafe_allow_html=True)
            if st.button("Execute Automated Screening", type="primary"):
                with st.spinner("Analyzing structural matrices..."):
                    classifier = tf.keras.models.load_model(r"models\classification\best_model_latest.keras")
                    
                    img_input = cv2.resize(img, (300, 300))
                    img_input = np.expand_dims(img_input, axis=0)
                    prob = classifier.predict(img_input, verbose=0)[0][0]
                    
                    st.write("---")
                    if prob <= 0.5:
                        confidence = (1 - prob) * 100
                        st.markdown("<div style='background-color: rgba(239, 68, 68, 0.2); border-left: 4px solid #ef4444; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                        st.error("⚠️ **Diagnostic Warning: Anomalous Tissue Patterns Detected**")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-box'><h5 style='color:#cbd5e1; margin:0;'>Anomaly Index Confidence</h5><h2 style='color:#ef4444; margin:0;'>{confidence:.2f}%</h2></div>", unsafe_allow_html=True)
                    else:
                        confidence = prob * 100
                        st.markdown("<div style='background-color: rgba(34, 197, 94, 0.2); border-left: 4px solid #22c55e; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                        st.success("✅ **Diagnostic Status: Unremarkable / Healthy Tissue Map**")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-box'><h5 style='color:#cbd5e1; margin:0;'>Structural Integrity Score</h5><h2 style='color:#22c55e; margin:0;'>{confidence:.2f}%</h2></div>", unsafe_allow_html=True)

elif app_mode == "🎯 Module Beta: Regional Boundary Localization":
    st.markdown("<h1>Module Beta: Regional Boundary Localization</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #06b6d4;'>Analysis Pipeline: High-Resolution Spatial Grid Mapping (HRBM-v1)</p>", unsafe_allow_html=True)
    st.write("---")
    
    uploaded_file = st.file_uploader("Drop targeted frame here or browse files...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown('<div class="clinical-card"><h4>Input Source Frame</h4></div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown('<div class="clinical-card"><h4>Localization Engine</h4></div>', unsafe_allow_html=True)
            if st.button("Delineate Region of Interest (ROI)", type="primary"):
                with st.spinner("Executing spatial boundary segmentation..."):
                    segmenter = YOLO(r"models\segmentation\best.pt")
                    
                    results = segmenter(img, conf=0.15, imgsz=512, retina_masks=True, verbose=False)[0]
                    
                    st.write("---")
                    if results.masks is not None:
                        annotated_img = results.plot()
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown("<div style='background-color: rgba(6, 182, 212, 0.2); border-left: 4px solid #06b6d4; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                        st.success("🎯 **Suspicious region coordinates successfully mapped and masked.**")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background-color: rgba(234, 179, 8, 0.2); border-left: 4px solid #eab308; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                        st.warning("⚠️ **No high-contrast anomaly boundaries thresholded by the spatial engine.**")
                        st.markdown("</div>", unsafe_allow_html=True)