"""
Utility Functions for SmartVision AI
Handles model loading and inference for both classification and detection
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import time

# ============================================
# CONFIGURATION
# ============================================

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'truck',
    'traffic light', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse',
    'cow', 'elephant', 'bottle', 'cup', 'bowl', 'pizza', 'cake',
    'chair', 'couch', 'bed', 'potted plant'
]

IMG_SIZE = (224, 224)

# Model paths
MODEL_PATHS = {
    'VGG16': 'models/VGG16_best.keras',
    'ResNet50': 'models/ResNet50_best.keras',
    'MobileNetV2': 'models/MobileNetV2_best.keras',
    'EfficientNetB0': 'models/EfficientNetB0_best.keras',
    'YOLOv8': 'runs/detect/smartvision_yolo/weights/best.pt'
}


# ============================================
# MODEL LOADING (WITH CACHING)
# ============================================

@tf.function
def predict_optimized(model, input_tensor):
    """Optimized prediction with TF function"""
    return model(input_tensor, training=False)


def load_classification_model(model_name):
    """
    Load a classification model
    
    Args:
        model_name: One of ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']
    
    Returns:
        Loaded Keras model
    """
    try:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Loading {model_name}...")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ {model_name} loaded successfully!")
        return model
    
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        return None


def load_yolo_model():
    """
    Load YOLOv8 detection model
    
    Returns:
        YOLO model object
    """
    try:
        model_path = MODEL_PATHS['YOLOv8']
        print("Loading YOLOv8...")
        model = YOLO(model_path)
        print("✅ YOLOv8 loaded successfully!")
        return model
    
    except Exception as e:
        print(f"❌ Error loading YOLOv8: {str(e)}")
        return None


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image_for_classification(image, target_size=IMG_SIZE):
    """
    Preprocess image for classification models
    
    Args:
        image: PIL Image or numpy array
        target_size: Tuple (width, height)
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(image)
    
    # Ensure RGB
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ============================================
# CLASSIFICATION INFERENCE
# ============================================

def predict_with_model(model, image, model_name, top_k=5):
    """
    Make prediction with a classification model
    
    Args:
        model: Loaded Keras model
        image: PIL Image
        model_name: Name of the model
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and inference time
    """
    try:
        # Start timer
        start_time = time.time()
        
        # Preprocess
        processed_image = preprocess_image_for_classification(image)
        
        # Predict
        predictions = model.predict(processed_image, verbose=0)
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get top-k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        top_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx]),
                'rank': rank + 1
            }
            for rank, idx in enumerate(top_indices)
        ]
        
        return {
            'model': model_name,
            'predictions': top_predictions,
            'inference_time_ms': round(inference_time, 2),
            'top_class': CLASS_NAMES[top_indices[0]],
            'top_confidence': float(predictions[0][top_indices[0]])
        }
    
    except Exception as e:
        print(f"❌ Prediction error with {model_name}: {str(e)}")
        return None


def predict_all_models(models_dict, image, top_k=5):
    """
    Get predictions from all classification models
    
    Args:
        models_dict: Dictionary of loaded models
        image: PIL Image
        top_k: Number of top predictions
    
    Returns:
        List of prediction results from all models
    """
    results = []
    
    for model_name, model in models_dict.items():
        if model is not None:
            result = predict_with_model(model, image, model_name, top_k)
            if result:
                results.append(result)
    
    return results


# ============================================
# OBJECT DETECTION INFERENCE
# ============================================

def predict_with_yolo(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Perform object detection with YOLOv8
    
    Args:
        model: YOLO model object
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Dictionary with detections and annotated image
    """
    try:
        # Start timer
        start_time = time.time()
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Predict
        results = model.predict(
            img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        # Extract results
        result = results[0]
        boxes = result.boxes
        
        # Parse detections
        detections = []
        for box in boxes:
            detection = {
                'class': CLASS_NAMES[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'bbox_normalized': box.xywhn[0].cpu().numpy().tolist()  # [x_center, y_center, width, height] normalized
            }
            detections.append(detection)
        
        # Get annotated image
        annotated_img = result.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_img)
        
        # Calculate statistics
        num_objects = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        fps = 1000 / inference_time if inference_time > 0 else 0
        
        return {
            'detections': detections,
            'annotated_image': annotated_pil,
            'num_objects': num_objects,
            'inference_time_ms': round(inference_time, 2),
            'fps': round(fps, 2),
            'avg_confidence': round(avg_confidence, 4)
        }
    
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return None


# ============================================
# VISUALIZATION HELPERS
# ============================================

def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes on image (alternative to YOLO's built-in)
    
    Args:
        image: PIL Image or numpy array
        detections: List of detection dictionaries
    
    Returns:
        PIL Image with bounding boxes
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Draw each detection
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Extract coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            img_array, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    return Image.fromarray(img_array)


def format_detection_table(detections):
    """
    Format detections for table display
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary suitable for st.table() or pandas DataFrame
    """
    table_data = {
        'Class': [],
        'Confidence': [],
        'Bounding Box': []
    }
    
    for det in detections:
        table_data['Class'].append(det['class'])
        table_data['Confidence'].append(f"{det['confidence']*100:.1f}%")
        bbox = det['bbox']
        table_data['Bounding Box'].append(f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]")
    
    return table_data


# ============================================
# MODEL COMPARISON
# ============================================

def compare_models(results_list):
    """
    Compare results from multiple classification models
    
    Args:
        results_list: List of prediction results from different models
    
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        'fastest_model': None,
        'slowest_model': None,
        'most_confident': None,
        'least_confident': None,
        'consensus_class': None
    }
    
    if not results_list:
        return comparison
    
    # Find fastest/slowest
    sorted_by_time = sorted(results_list, key=lambda x: x['inference_time_ms'])
    comparison['fastest_model'] = sorted_by_time[0]['model']
    comparison['slowest_model'] = sorted_by_time[-1]['model']
    
    # Find most/least confident
    sorted_by_conf = sorted(results_list, key=lambda x: x['top_confidence'])
    comparison['least_confident'] = sorted_by_conf[0]['model']
    comparison['most_confident'] = sorted_by_conf[-1]['model']
    
    # Find consensus (most common top prediction)
    top_classes = [r['top_class'] for r in results_list]
    comparison['consensus_class'] = max(set(top_classes), key=top_classes.count)
    
    return comparison


# ============================================
# ERROR HANDLING
# ============================================

def validate_image(image):
    """
    Validate uploaded image
    
    Args:
        image: PIL Image or file object
    
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        if image is None:
            return False, "No image provided"
        
        # Check if PIL Image
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.open(image)
        
        # Check size
        if img.size[0] < 50 or img.size[1] < 50:
            return False, "Image too small (minimum 50×50 pixels)"
        
        if img.size[0] > 5000 or img.size[1] > 5000:
            return False, "Image too large (maximum 5000×5000 pixels)"
        
        # Check format
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False, f"Unsupported image mode: {img.mode}"
        
        return True, None
    
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("SmartVision AI Utilities")
    print("=" * 50)
    print("\nAvailable Functions:")
    print("  - load_classification_model(model_name)")
    print("  - load_yolo_model()")
    print("  - predict_with_model(model, image, model_name)")
    print("  - predict_with_yolo(model, image, conf, iou)")
    print("  - compare_models(results)")
    print("\nSupported Classes:")
    for i, cls in enumerate(CLASS_NAMES, 1):
        print(f"  {i:2d}. {cls}")