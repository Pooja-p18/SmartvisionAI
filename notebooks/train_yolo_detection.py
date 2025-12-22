"""
YOLOv8 Object Detection Training
Train on 25-class COCO subset for multi-object detection

Purpose: Detect and locate multiple objects in images
"""

from ultralytics import YOLO
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_YAML = "smartvision_dataset/detection/data.yaml"
MODEL_SIZE = "yolov8n"  # Options: n (nano), s (small), m (medium), l (large), x (extra)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16

# ============================================
# VERIFY DATASET
# ============================================
def verify_dataset():
    """Check if dataset is properly formatted"""
    
    print("🔍 Verifying dataset structure...")
    
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    base_path = data_config['path']
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    
    # Count files
    images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    print(f"✅ Found {len(images)} images")
    print(f"✅ Found {len(labels)} label files")
    print(f"✅ Classes: {data_config['nc']}")
    print(f"✅ Class names: {list(data_config['names'].values())[:5]}... (+{data_config['nc']-5} more)")
    
    return data_config


# ============================================
# TRAIN YOLO MODEL
# ============================================
def train_yolo():
    """
    Train YOLOv8 on custom dataset
    
    Process:
    1. Load pre-trained YOLO model
    2. Fine-tune on our 25 classes
    3. Save best weights
    """
    
    print("\n" + "="*70)
    print("🚀 TRAINING YOLOV8")
    print("="*70)
    
    # Load pre-trained model
    print(f"\n📥 Loading {MODEL_SIZE}.pt (pre-trained on COCO)...")
    model = YOLO(f'{MODEL_SIZE}.pt')
    
    print("\n🎯 Starting training...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    # Train
    results = model.train(
     data=DATA_YAML,
     epochs=EPOCHS,
     imgsz=IMG_SIZE,
     batch=BATCH_SIZE,
     name='smartvision_yolo',
     patience=20,
     save=True,
     device='cpu',          # ✅ FIXED
     workers=4,
     pretrained=True,
     optimizer='Adam',
     lr0=0.001,
     lrf=0.01,
     momentum=0.9,
     weight_decay=0.0005,
     warmup_epochs=3,
     cos_lr=True,
     label_smoothing=0.0,
     box=7.5,
     cls=0.5,
     dfl=1.5,
     plots=True,
     verbose=True
   )

    
    print("\n✅ Training complete!")
    return model, results


# ============================================
# EVALUATE MODEL
# ============================================
def evaluate_yolo(model):
    """Evaluate trained model on validation set"""
    
    print("\n" + "="*70)
    print("📊 EVALUATING MODEL")
    print("="*70)
    
    # Validate
    results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,                # Confidence threshold
        iou=0.6,                  # IoU threshold for NMS
        device=0
    )
    
    # Print metrics
    print("\n📈 Performance Metrics:")
    print(f"   mAP@0.5:      {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision:    {results.box.p:.4f}")
    print(f"   Recall:       {results.box.r:.4f}")
    
    return results


# ============================================
# TEST ON SAMPLE IMAGES
# ============================================
def test_predictions(model, num_samples=6):
    """Test model on sample images"""
    
    print("\n" + "="*70)
    print("🖼️  TESTING ON SAMPLE IMAGES")
    print("="*70)
    
    # Get sample images
    images_path = "smartvision_dataset/detection/images"
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')][:num_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_path, img_file)
        
        # Predict
        results = model.predict(
            source=img_path,
            conf=0.25,
            iou=0.6,
            show=False,
            save=False
        )
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Count detections
        num_detections = len(results[0].boxes)
        
        # Display
        axes[idx].imshow(annotated_img)
        axes[idx].set_title(f'{img_file}\n{num_detections} objects detected', fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/yolo_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Sample predictions saved to 'results/yolo_predictions.png'")


# ============================================
# VISUALIZE TRAINING RESULTS
# ============================================
def visualize_training_results():
    """Plot training metrics from YOLOv8 logs"""
    
    print("\n📊 Visualizing training results...")
    
    # YOLO saves results in runs/detect/smartvision_yolo/
    results_dir = Path('runs/detect/smartvision_yolo')
    
    if not results_dir.exists():
        print("⚠️  Training results directory not found")
        return
    
    # Load results CSV
    results_csv = results_dir / 'results.csv'
    
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # mAP
        axes[0, 0].plot(df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2)
        axes[0, 0].plot(df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mAP')
        axes[0, 0].set_title('Mean Average Precision')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Precision & Recall
        axes[0, 1].plot(df['metrics/precision(B)'], label='Precision', linewidth=2)
        axes[0, 1].plot(df['metrics/recall(B)'], label='Recall', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision & Recall')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Losses
        axes[1, 0].plot(df['train/box_loss'], label='Box Loss', linewidth=2)
        axes[1, 0].plot(df['train/cls_loss'], label='Class Loss', linewidth=2)
        axes[1, 0].plot(df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Validation Loss
        axes[1, 1].plot(df['val/box_loss'], label='Val Box Loss', linewidth=2)
        axes[1, 1].plot(df['val/cls_loss'], label='Val Class Loss', linewidth=2)
        axes[1, 1].plot(df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Validation Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/yolo_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training metrics visualization saved!")


# ============================================
# EXPORT MODEL
# ============================================
def export_model(model):
    """Export model for deployment"""
    
    print("\n" + "="*70)
    print("📦 EXPORTING MODEL")
    print("="*70)
    
    # Export to different formats
    formats = ['torchscript', 'onnx']  # Add more: 'tflite', 'coreml', etc.
    
    for fmt in formats:
        try:
            print(f"\n📤 Exporting to {fmt.upper()}...")
            model.export(format=fmt, imgsz=IMG_SIZE)
            print(f"✅ {fmt.upper()} export successful!")
        except Exception as e:
            print(f"❌ {fmt.upper()} export failed: {str(e)}")
    
    print("\n✅ Model export complete!")
    print(f"📁 Exported models saved in: runs/detect/smartvision_yolo/weights/")


# ============================================
# MAIN PIPELINE
# ============================================
def main():
    """Complete YOLO training pipeline"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*70)
    print("🎯 YOLOV8 OBJECT DETECTION TRAINING PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Verify dataset structure")
    print("  2. Train YOLOv8 model")
    print("  3. Evaluate performance")
    print("  4. Test on sample images")
    print("  5. Export model for deployment")
    print()
    
    # Step 1: Verify dataset
    data_config = verify_dataset()
    
    # Step 2: Train model
    model, train_results = train_yolo()
    
    # Step 3: Evaluate
    val_results = evaluate_yolo(model)
    
    # Step 4: Test predictions
    test_predictions(model)
    
    # Step 5: Visualize results
    visualize_training_results()
    
    # Step 6: Export model
    export_model(model)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ YOLO TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Performance:")
    print(f"   mAP@0.5:      {val_results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {val_results.box.map:.4f}")
    print(f"   Precision:    {val_results.box.p:.4f}")
    print(f"   Recall:       {val_results.box.r:.4f}")
    print(f"\n📁 Model weights: runs/detect/smartvision_yolo/weights/best.pt")
    print(f"📁 Training logs: runs/detect/smartvision_yolo/")
    print(f"📊 Visualizations: results/")


if __name__ == "__main__":
    main()