---
title: SmartvisionAI
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Intelligent Multi-Class Object Recognition System
license: mit
---

## Features
- Image classification
- Object detection
- Interactive UI

## Tech Stack
- Python
- Streamlit
- OpenCV
- YOLO
- Hugging Face Spaces

# 👁️ SmartVision AI  
### Image Classification & Object Detection using Deep Learning

SmartVision AI is an end-to-end **computer vision project** that performs **image classification** and **object detection** using state-of-the-art deep learning models.  
The project is deployed as an **interactive web application using Streamlit** and hosted on **Hugging Face Spaces**.

---

## 🚀 Features

### 🖼️ Image Classification
- Classifies images into **25 object categories**
- Built using **MobileNetV2 (Transfer Learning)**
- Optimized for **CPU-based inference**

### 📦 Object Detection
- Detects and localizes **multiple objects** in a single image
- Uses **YOLOv8 (Ultralytics)**
- Outputs bounding boxes with class labels & confidence scores

### 🌐 Web Application
- Interactive UI built with **Streamlit**
- Upload images directly and view predictions instantly
- Deployed online via **Hugging Face Spaces**

---

## 🧠 Models Used

| Task | Model | Description |
|-----|------|-------------|
| Classification | MobileNetV2 | Lightweight CNN with transfer learning |
| Detection | YOLOv8n | Fast and efficient object detector |

---

## 📊 Dataset

- COCO-style custom dataset
- **25 object classes**
- Dataset structure follows YOLOv8 standards

**Classes include:**  
person, bicycle, car, dog, cat, chair, bottle, bus, truck, airplane, etc.

---

## 🗂️ Project Structure

SmartVisionAI/
│
├── src/
│ ├── streamlit_app.py # Main Streamlit application
│ └── utils.py # Helper functions (model loading, preprocessing)
│
├── notebooks/ # EDA & training notebooks/scripts
├── tests/ # Testing scripts
├── models/ # Saved model weights
├── results/ # Plots and metrics
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore
