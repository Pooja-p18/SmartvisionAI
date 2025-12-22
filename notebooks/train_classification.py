"""
CPU-Optimized Transfer Learning for Image Classification
Model: MobileNetV2
Purpose: 25-class image classification on local laptop
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ============================================
# CONFIGURATION (CPU SAFE)
# ============================================
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 25
LEARNING_RATE = 0.0005

BASE_DIR = "smartvision_dataset/classification"
TRAIN_DIR = f"{BASE_DIR}/train"
VAL_DIR = f"{BASE_DIR}/val"
TEST_DIR = f"{BASE_DIR}/test"

# ============================================
# DATA GENERATORS
# ============================================
print("\n📁 Preparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("✅ Data loaded successfully")

# ============================================
# MODEL BUILDING
# ============================================
def build_model():
    print("\n🏗️ Building MobileNetV2 model...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base_model.trainable = False  # freeze base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model built with {model.count_params():,} parameters")
    return model, base_model

# ============================================
# TRAINING
# ============================================
def train_model(model, base_model):
    print("\n🚀 Starting training...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            "models/mobilenet_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n📚 Phase 1: Training top layers (5 epochs)")
    history1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    print("\n📚 Phase 2: Fine-tuning entire model (10 epochs)")
    base_model.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        initial_epoch=5,
        callbacks=callbacks,
        verbose=1
    )

    history = {
        "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
        "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
        "loss": history1.history["loss"] + history2.history["loss"],
        "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
    }

    return model, history

# ============================================
# EVALUATION
# ============================================
def evaluate_model(model):
    print("\n📊 Evaluating model on test set...")

    loss, acc = model.evaluate(test_generator, verbose=1)

    print("\n" + "="*40)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("="*40)

    return acc, loss

# ============================================
# PLOT HISTORY
# ============================================
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=300)
    plt.close()

# ============================================
# MAIN
# ============================================
def main():
    model, base_model = build_model()
    model, history = train_model(model, base_model)
    plot_history(history)
    acc, loss = evaluate_model(model)

    model.save("models/mobilenet_final.keras")

    with open("results/final_results.json", "w") as f:
        json.dump({"accuracy": float(acc), "loss": float(loss)}, f, indent=2)

    print("\n✅ TRAINING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    print("💻 Running on CPU (Laptop mode)")
    main()
