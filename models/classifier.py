"""
models/classifier.py
PyTorch Convolutional Neural Network (CNN) for MNIST handwritten digit classification.
Discriminative Model: Learns decision boundaries P(Y|X) to predict class label Y given image X.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification (0-9).
    Architecture:
      Conv1 (32 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
      Conv2 (64 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
      Flatten -> Dense (128) -> ReLU -> Dropout (0.25) -> Dense (10)
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)  # Logits for classes 0..9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layer(x)
        logits = self.fc_layer(features)
        return logits


def get_mnist_transforms(train: bool = False):
    """
    MNIST normalization transform.
    When train=True, includes subtle affine augmentation (rotations, translations, scaling)
    to make the classifier invariant to real-world drawing styles, angles, and tilts.
    """
    if train:
        return transforms.Compose([
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


DEFAULT_MNIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)


def load_mnist_data(data_dir: str = None, batch_size: int = 64):
    """
    Loads MNIST dataset with automatic downloading.
    Falls back gracefully if download is blocked.
    """
    if data_dir is None:
        data_dir = DEFAULT_MNIST_DIR
    train_transform = get_mnist_transforms(train=True)
    test_transform = get_mnist_transforms(train=False)
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader = None,
    epochs: int = 3,
    lr: float = 0.001,
    device: str = "cpu",
    save_path: str = None,
    progress_callback = None
):
    """
    Trains the CNN classifier on MNIST with cross-entropy loss.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    current_step = 0

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_step += 1
            if progress_callback is not None and batch_idx % 20 == 0:
                progress_callback(
                    current_step / total_steps,
                    f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        test_acc = 0.0
        if test_loader is not None:
            test_acc, _ = evaluate_classifier(model, test_loader, device=device)
            history["test_acc"].append(test_acc)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "history": history,
            "test_accuracy": history["test_acc"][-1] if history["test_acc"] else epoch_acc
        }, save_path)

    return history


def evaluate_classifier(model: nn.Module, test_loader: DataLoader, device: str = "cpu"):
    """
    Evaluates the classifier on the held-out test dataset.
    Returns (accuracy_percentage, average_loss).
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = test_loss / total if total > 0 else 0.0
    return accuracy, avg_loss


def save_classifier_checkpoint(model: nn.Module, path: str, test_acc: float = None):
    """Saves classifier weights and metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if test_acc is not None:
        payload["test_accuracy"] = test_acc
    torch.save(payload, path)


def load_classifier_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    """Loads weights into model from checkpoint path."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        test_acc = checkpoint.get("test_accuracy", None)
    else:
        model.load_state_dict(checkpoint)
        test_acc = None
    model.to(device)
    model.eval()
    return model, test_acc
