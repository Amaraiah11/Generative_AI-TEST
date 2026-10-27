"""
finetune_cnn.py
Fine-tunes the CNN Classifier on MNIST with data augmentation to enhance real-world drawing accuracy.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from backend.inference import get_device, DEFAULT_CLASSIFIER_PATH
from models.classifier import DigitCNN, train_classifier, load_mnist_data, load_classifier_checkpoint, save_classifier_checkpoint, evaluate_classifier

device = get_device()
print(f"[INFO] Fine-tuning CNN on {device} with data augmentation...")

model = DigitCNN().to(device)
if os.path.exists(DEFAULT_CLASSIFIER_PATH):
    model, _ = load_classifier_checkpoint(model, DEFAULT_CLASSIFIER_PATH, device=str(device))
    print("Loaded existing weights.")

train_loader, test_loader = load_mnist_data(batch_size=64)

def progress_cb(pct, msg):
    if int(pct * 100) % 20 == 0:
        print(f"[Fine-tuning] {int(pct*100)}%: {msg}")

history = train_classifier(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=3,
    lr=0.0005,
    device=str(device),
    save_path=DEFAULT_CLASSIFIER_PATH,
    progress_callback=progress_cb
)

final_acc, _ = evaluate_classifier(model, test_loader, device=str(device))
save_classifier_checkpoint(model, DEFAULT_CLASSIFIER_PATH, test_acc=final_acc)
print(f"[SUCCESS] Fine-tuning complete! Final Test Accuracy: {final_acc:.2f}%")
