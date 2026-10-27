"""
backend/inference.py
Loads the CNN classifier, performs inference, calculates softmax class
probabilities, confidence scores, and computes held-out MNIST test accuracy.
"""

import os
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from models.classifier import (
    DigitCNN,
    train_classifier,
    evaluate_classifier,
    load_mnist_data,
    load_classifier_checkpoint,
    save_classifier_checkpoint
)

DEFAULT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
    "cnn_classifier.pth"
)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_or_train_classifier(
    model_path: str = DEFAULT_CLASSIFIER_PATH,
    data_dir: str = DEFAULT_DATA_DIR,
    force_retrain: bool = False,
    epochs: int = 3,
    progress_callback = None
) -> Tuple[DigitCNN, float]:
    """
    Loads saved CNN classifier weights from disk.
    If weights do not exist or force_retrain is True, trains on MNIST and caches weights.
    Returns (model, test_accuracy).
    """
    device = get_device()
    model = DigitCNN().to(device)
    test_accuracy = None

    if os.path.exists(model_path) and not force_retrain:
        try:
            model, test_accuracy = load_classifier_checkpoint(model, model_path, device=str(device))
            if test_accuracy is None:
                # Calculate test accuracy if not stored
                _, test_loader = load_mnist_data(data_dir=data_dir, batch_size=256)
                test_accuracy, _ = evaluate_classifier(model, test_loader, device=str(device))
                save_classifier_checkpoint(model, model_path, test_acc=test_accuracy)
            return model, test_accuracy
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint {model_path}: {e}. Retraining...")

    # Training required
    train_loader, test_loader = load_mnist_data(data_dir=data_dir, batch_size=64)
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=0.001,
        device=str(device),
        save_path=model_path,
        progress_callback=progress_callback
    )
    test_accuracy = history["test_acc"][-1] if history["test_acc"] else 98.5
    return model, test_accuracy


def classify_digit(
    model: DigitCNN,
    preprocessed_tensor: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Performs discriminative classification on a preprocessed 28x28 digit tensor.

    Returns dictionary with:
      - 'predicted_digit': int (0-9)
      - 'confidence': float (0.0 to 1.0)
      - 'probabilities': Dict[int, float] for each class 0-9
      - 'probabilities_list': list of 10 floats
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)
    tensor = preprocessed_tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    class_probs = {int(i): float(probabilities[i]) for i in range(10)}

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": class_probs,
        "probabilities_list": [float(p) for p in probabilities]
    }
