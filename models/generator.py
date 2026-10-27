"""
models/generator.py
PyTorch Variational Autoencoder (VAE) for MNIST handwritten digit generation.
Generative Model: Learns the underlying data distribution P(X) via latent variable model P(X|Z)P(Z).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for 28x28 handwritten digit generation.
    - Latent dimension: 20
    - Encoder: Conv2D layers -> Dense -> mu and logvar
    - Reparameterization trick: z = mu + eps * exp(0.5 * logvar)
    - Decoder: Dense -> ConvTranspose2D layers -> Sigmoid output in [0, 1]
    """
    def __init__(self, latent_dim: int = 20):
        super(DigitVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: latent_dim -> 128 -> 64*7*7 -> 32*14*14 -> 1*28*28
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid()  # Pixel intensity strictly in [0, 1]
        )

    def encode(self, x: torch.Tensor):
        features = self.encoder_conv(x)
        h = self.encoder_fc(features)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: allows gradients to backpropagate through latent sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 7)
        recon_x = self.decoder_conv(h)
        return recon_x

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def sample(self, num_samples: int = 10, device: str = "cpu", temperature: float = 1.0) -> torch.Tensor:
        """
        Samples latent vectors z ~ N(0, (temperature)^2 * I) and decodes them into synthetic images.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
            samples = self.decode(z)
        return samples


def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    """
    VAE Objective: ELBO (Evidence Lower Bound) = Reconstruction Loss + KL Divergence
    - Reconstruction Loss: Binary Cross-Entropy between generated and target pixels.
    - KL Divergence: D_KL(q(z|x) || p(z)) where p(z) ~ N(0, I).
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # KL divergence closed form for Gaussian: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld


def get_vae_mnist_transforms():
    """Transforms for VAE: pixel intensities normalized strictly to [0.0, 1.0]."""
    return transforms.Compose([
        transforms.ToTensor()
    ])


DEFAULT_MNIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)


def load_vae_mnist_data(data_dir: str = None, batch_size: int = 64):
    """Loads MNIST data with [0, 1] normalization for VAE training."""
    if data_dir is None:
        data_dir = DEFAULT_MNIST_DIR
    transform = get_vae_mnist_transforms()
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    device: str = "cpu",
    save_path: str = None,
    progress_callback = None
):
    """Trains the Variational Autoencoder on MNIST images."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    current_step = 0
    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        running_bce = 0.0
        running_kld = 0.0
        total_samples = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            recon_images, mu, logvar = model(images)
            loss, bce, kld = vae_loss_function(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_total += loss.item()
            running_bce += bce.item()
            running_kld += kld.item()
            total_samples += batch_size

            current_step += 1
            if progress_callback is not None and batch_idx % 20 == 0:
                progress_callback(
                    current_step / total_steps,
                    f"VAE Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item() / batch_size:.1f}"
                )

        history["total_loss"].append(running_total / total_samples)
        history["recon_loss"].append(running_bce / total_samples)
        history["kl_loss"].append(running_kld / total_samples)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "history": history,
            "latent_dim": getattr(model, "latent_dim", 20)
        }, save_path)

    return history


def save_vae_checkpoint(model: nn.Module, path: str):
    """Saves VAE weights."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": getattr(model, "latent_dim", 20)
    }, path)


def load_vae_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    """Loads VAE weights."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
