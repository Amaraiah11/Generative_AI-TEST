"""
train_all.py
Standalone training utility to pre-train both the CNN Classifier and the VAE Generator
on the MNIST dataset and save weights into saved_models/.
"""

import os
import sys
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from backend.inference import load_or_train_classifier
from backend.generation import load_or_train_vae, generate_digit_samples
from models.classifier import DigitCNN, evaluate_classifier, load_mnist_data
from models.generator import DigitVAE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Model Training Pipeline on {device} ===")

    saved_models_dir = os.path.join(BASE_DIR, "saved_models")
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(saved_models_dir, exist_ok=True)

    cnn_path = os.path.join(saved_models_dir, "cnn_classifier.pth")
    vae_path = os.path.join(saved_models_dir, "vae_generator.pth")

    def cnn_progress(pct, msg):
        print(f"[CNN] {int(pct*100)}%: {msg}")

    def vae_progress(pct, msg):
        print(f"[VAE] {int(pct*100)}%: {msg}")

    start_time = time.time()

    print("\n--- 1. Training Discriminative CNN Classifier ---")
    classifier_model, test_acc = load_or_train_classifier(
        model_path=cnn_path,
        data_dir=data_dir,
        force_retrain=True,
        epochs=3,
        progress_callback=cnn_progress
    )
    print(f"CNN Classifier Training Complete! Held-out Test Accuracy: {test_acc:.2f}%")
    print(f"Saved weights to: {cnn_path}")

    print("\n--- 2. Training Generative Variational Autoencoder (VAE) ---")
    vae_model = load_or_train_vae(
        model_path=vae_path,
        data_dir=data_dir,
        force_retrain=True,
        epochs=4,
        progress_callback=vae_progress
    )
    print("VAE Generator Training Complete!")
    print(f"Saved weights to: {vae_path}")

    print("\n--- 3. Verifying Generation Inference ---")
    samples = generate_digit_samples(vae_model, num_samples=5, device=device)
    print(f"Generated {len(samples)} synthetic digit samples successfully!")

    total_time = time.time() - start_time
    print(f"\nAll models successfully trained and verified in {total_time:.1f} seconds!")


if __name__ == "__main__":
    main()
