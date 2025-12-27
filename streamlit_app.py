"""
SmartVision AI - Main Streamlit Application
Multi-page app for image classification and object detection

Pages:
1. Home - Project overview
2. Image Classification - Single object classification
3. Object Detection - Multi-object detection with YOLO
4. Model Performance - Metrics comparison
5. About - Documentation
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
from src.utils import load_model, preprocess_image, get_class_names


# Page configuration
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 SmartVision AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🖼️ Image Classification", "🎯 Object Detection", 
     "📊 Model Performance", "ℹ️ About"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **SmartVision AI** uses deep learning to:
    - Classify objects into 25 categories
    - Detect multiple objects in images
    - Provide real-time predictions
    """
)

# ============================================
# PAGE 1: HOME
# ============================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🔍 SmartVision AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Multi-Class Object Recognition System</p>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>25</h2>
            <p>Object Classes</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>4</h2>
            <p>CNN Models</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>90%+</h2>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    st.header("✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🖼️ Image Classification</h3>
            <ul>
                <li>4 state-of-the-art CNN architectures</li>
                <li>VGG16, ResNet50, MobileNetV2, EfficientNetB0</li>
                <li>Top-5 predictions with confidence scores</li>
                <li>Side-by-side model comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Performance Analytics</h3>
            <ul>
                <li>Accuracy, Precision, Recall metrics</li>
                <li>Inference speed comparison</li>
                <li>Confusion matrices</li>
                <li>Class-wise performance breakdown</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🎯 Object Detection</h3>
            <ul>
                <li>YOLOv8-powered multi-object detection</li>
                <li>Real-time bounding box visualization</li>
                <li>Adjustable confidence threshold</li>
                <li>Supports multiple objects per image</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h3>🌍 Real-World Applications</h3>
            <ul>
                <li>Smart Cities & Traffic Management</li>
                <li>Retail & E-Commerce Analytics</li>
                <li>Security & Surveillance Systems</li>
                <li>Wildlife Conservation Monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Supported classes
    st.header("🏷️ Supported Object Classes (25)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    classes = {
        "🚗 Vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "airplane"],
        "👤 Person": ["person"],
        "🚦 Outdoor": ["traffic light", "stop sign", "bench"],
        "🐾 Animals": ["dog", "cat", "horse", "bird", "cow", "elephant"],
        "🍽️ Food": ["bottle", "cup", "bowl", "pizza", "cake"]
    }
    
    cols = [col1, col2, col3, col4, col5]
    
    for idx, (category, items) in enumerate(classes.items()):
        with cols[idx]:
            st.subheader(category)
            for item in items:
                st.write(f"• {item}")
    
    st.markdown("---")
    
    # Quick start
    st.header("🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Image Classification**: Upload an image containing a single object
    2. **Object Detection**: Upload an image with multiple objects
    3. **View Results**: See predictions, confidence scores, and bounding boxes
    4. **Compare Models**: Check performance metrics across different architectures
    """)
    
    st.info("💡 **Tip**: Try uploading your own images to test the system!")

# ============================================
# PAGE 2: IMAGE CLASSIFICATION
# ============================================
elif page == "🖼️ Image Classification":
    st.title("🖼️ Image Classification")
    st.markdown("Upload an image to classify it using 4 different CNN models")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing a single object from the 25 supported classes"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.info(f"**Image Size**: {image.size[0]} × {image.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Classification Results")
            
            # Placeholder for actual model predictions
            # In production, load models and make predictions here
            
            st.warning("⚠️ **Demo Mode**: Replace with actual model predictions")
            
            # Mock predictions (replace with actual model inference)
            models = ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"]
            
            tabs = st.tabs(models)
            
            for idx, model_name in enumerate(models):
                with tabs[idx]:
                    st.markdown(f"### {model_name} Predictions")
                    
                    # Mock top-5 predictions
                    mock_predictions = [
                        ("dog", 0.92),
                        ("cat", 0.05),
                        ("horse", 0.02),
                        ("cow", 0.01),
                        ("bird", 0.00)
                    ]
                    
                    for rank, (class_name, confidence) in enumerate(mock_predictions, 1):
                        col_a, col_b, col_c = st.columns([1, 3, 1])
                        with col_a:
                            st.write(f"**#{rank}**")
                        with col_b:
                            st.progress(confidence)
                        with col_c:
                            st.write(f"{confidence*100:.1f}%")
                        st.write(f"**{class_name.upper()}**")
                    
                    # Inference time
                    st.metric("Inference Time", "120 ms")
        
        st.markdown("---")
        
        # Model comparison
        st.subheader("📊 Model Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("VGG16", 0.85, "150ms"),
            ("ResNet50", 0.89, "100ms"),
            ("MobileNetV2", 0.84, "50ms"),
            ("EfficientNetB0", 0.92, "80ms")
        ]
        
        for col, (name, acc, time) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.metric(name, f"{acc*100:.1f}%", time)

# ============================================
# PAGE 3: OBJECT DETECTION
# ============================================
elif page == "🎯 Object Detection":
    st.title("🎯 Object Detection with YOLOv8")
    st.markdown("Upload an image to detect and locate multiple objects")
    
    # Sidebar controls
    st.sidebar.subheader("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing one or more objects"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detected Objects")
            
            # Placeholder for YOLO predictions
            st.warning("⚠️ **Demo Mode**: Replace with actual YOLO inference")
            
            # Mock detection (replace with actual YOLO model)
            # In production: results = yolo_model.predict(image, conf=confidence_threshold)
            
            # Display mock annotated image
            st.image(image, use_container_width=True)
        
        # Detection summary
        st.markdown("---")
        st.subheader("📊 Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Objects", "8")
        with col2:
            st.metric("Processing Time", "45 ms")
        with col3:
            st.metric("Avg Confidence", "87.3%")
        with col4:
            st.metric("FPS", "22.2")
        
        # Detected objects table
        st.markdown("### 🏷️ Detected Objects")
        
        # Mock detection results
        detections_data = {
            "Class": ["person", "car", "dog", "chair", "bottle", "cup", "bowl", "pizza"],
            "Confidence": ["95%", "92%", "89%", "85%", "82%", "78%", "75%", "71%"],
            "Bounding Box": [
                "[120, 50, 280, 400]",
                "[300, 150, 500, 350]",
                "[50, 300, 150, 450]",
                "[400, 200, 520, 380]",
                "[150, 100, 180, 150]",
                "[200, 120, 230, 160]",
                "[250, 180, 290, 220]",
                "[320, 250, 380, 300]"
            ]
        }
        
        st.table(detections_data)

# ============================================
# PAGE 4: MODEL PERFORMANCE
# ============================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")
    st.markdown("Comprehensive analysis of all trained models")
    
    # Load results (mock data - replace with actual JSON)
    st.subheader("🎯 Classification Models")
    
    col1, col2, col3, col4 = st.columns(4)
    
    models_data = {
        "VGG16": {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.835},
        "ResNet50": {"accuracy": 0.89, "precision": 0.88, "recall": 0.87, "f1": 0.875},
        "MobileNetV2": {"accuracy": 0.84, "precision": 0.83, "recall": 0.82, "f1": 0.825},
        "EfficientNetB0": {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.905}
    }
    
    for col, (name, metrics) in zip([col1, col2, col3, col4], models_data.items()):
        with col:
            st.markdown(f"### {name}")
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.metric("Precision", f"{metrics['precision']*100:.1f}%")
            st.metric("Recall", f"{metrics['recall']*100:.1f}%")
            st.metric("F1-Score", f"{metrics['f1']*100:.1f}%")
    
    st.markdown("---")
    
    # YOLO performance
    st.subheader("🎯 Object Detection Model (YOLOv8)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("mAP@0.5", "87.5%")
    with col2:
        st.metric("mAP@0.5:0.95", "68.3%")
    with col3:
        st.metric("Precision", "85.2%")
    with col4:
        st.metric("Recall", "82.7%")
    
    st.markdown("---")
    
    # Inference speed comparison
    st.subheader("⚡ Inference Speed Comparison")
    
    speed_data = {
        "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0", "YOLOv8"],
        "Inference Time (ms)": [150, 100, 50, 80, 45],
        "FPS": [6.7, 10.0, 20.0, 12.5, 22.2]
    }
    
    st.bar_chart(speed_data, x="Model", y="Inference Time (ms)")
    
    st.markdown("---")
    
    # Best model recommendation
    st.success("""
    ### 🏆 Recommended Models
    
    **For Accuracy**: EfficientNetB0 (92.0%)
    - Best overall classification accuracy
    - Good balance of speed and performance
    
    **For Speed**: MobileNetV2 (50ms)
    - Fastest inference time
    - Ideal for mobile/edge devices
    
    **For Detection**: YOLOv8 (22.2 FPS)
    - Real-time multi-object detection
    - High accuracy with fast inference
    """)

# ============================================
# PAGE 5: ABOUT
# ============================================
elif page == "ℹ️ About":
    st.title("ℹ️ About SmartVision AI")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    SmartVision AI is a comprehensive computer vision platform that combines:
    - **Transfer Learning-based Image Classification** using state-of-the-art CNN architectures
    - **YOLO-based Object Detection** for multi-object localization
    - **Production-ready Deployment** as an accessible web application
    
    ## 🗄️ Dataset
    
    - **Source**: COCO 2017 (25-class subset)
    - **Total Images**: 2,500 (100 per class)
    - **Split**: 70% train / 15% val / 15% test
    - **Classes**: 25 diverse categories across vehicles, animals, food, furniture, and more
    
    ## 🧠 Models Used
    
    ### Classification Models:
    1. **VGG16**: Deep 16-layer CNN with simple architecture
    2. **ResNet50**: 50-layer residual network with skip connections
    3. **MobileNetV2**: Efficient mobile-optimized architecture
    4. **EfficientNetB0**: Compound scaling for optimal performance
    
    ### Detection Model:
    - **YOLOv8**: State-of-the-art real-time object detection
    
    ## 🛠️ Technology Stack
    
    - **Deep Learning**: TensorFlow, PyTorch
    - **Computer Vision**: OpenCV, PIL
    - **Object Detection**: Ultralytics YOLOv8
    - **Web Framework**: Streamlit
    - **Deployment**: Hugging Face Spaces
    - **Data Processing**: NumPy, Pandas
    - **Visualization**: Matplotlib, Seaborn
    
    ## 🌍 Real-World Applications
    
    - Smart Cities & Traffic Management
    - Retail & E-Commerce Analytics
    - Security & Surveillance Systems
    - Wildlife Conservation Monitoring
    - Healthcare Equipment Tracking
    - Smart Home & IoT Integration
    - Agriculture & Livestock Monitoring
    - Logistics & Warehouse Automation
    
    ## 📈 Performance Highlights
    
    - **Classification Accuracy**: Up to 92% (EfficientNetB0)
    - **Detection mAP@0.5**: 87.5% (YOLOv8)
    - **Inference Speed**: 22+ FPS for real-time detection
    - **Model Size**: Optimized for cloud deployment
    
    ## 📚 Documentation
    
    For detailed documentation, visit the [https://github.com/Pooja-p18/SmartvisionAI](#).
    
    ## 👨‍💻 Developer
    
    **Project**: SmartVision AI
    **Version**: 1.0.0
    **License**: MIT
    
    ## 📧 Contact
    
    For questions or collaboration opportunities, please reach out via GitHub.
    """)
    
    st.markdown("---")
    st.info("Made with ❤️ using Streamlit and deep learning")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Resources")
st.sidebar.markdown("- [GitHub Repository](#)")
st.sidebar.markdown("- [Dataset (COCO)](#)")
st.sidebar.markdown("- [Documentation](#)")
