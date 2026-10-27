"""
backend/generation.py
Loads the Variational Autoencoder (VAE), generates new handwritten digit samples
by sampling from the continuous latent space, and provides reconstruction / interpolation.
"""

import os
from typing import Tuple, List, Optional
import torch
import numpy as np
from PIL import Image

from models.generator import (
    DigitVAE,
    train_vae,
    load_vae_mnist_data,
    load_vae_checkpoint,
    save_vae_checkpoint
)

DEFAULT_VAE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
    "vae_generator.pth"
)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_or_train_vae(
    model_path: str = DEFAULT_VAE_PATH,
    data_dir: str = DEFAULT_DATA_DIR,
    force_retrain: bool = False,
    latent_dim: int = 20,
    epochs: int = 5,
    progress_callback = None
) -> DigitVAE:
    """
    Loads saved VAE weights from disk.
    If weights do not exist or force_retrain is True, trains on MNIST and caches weights.
    Returns the loaded VAE model.
    """
    device = get_device()
    model = DigitVAE(latent_dim=latent_dim).to(device)

    if os.path.exists(model_path) and not force_retrain:
        try:
            model = load_vae_checkpoint(model, model_path, device=str(device))
            return model
        except Exception as e:
            print(f"[Warning] Failed to load VAE checkpoint {model_path}: {e}. Retraining...")

    # Training required
    train_loader, _ = load_vae_mnist_data(data_dir=data_dir, batch_size=64)
    train_vae(
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        lr=0.001,
        device=str(device),
        save_path=model_path,
        progress_callback=progress_callback
    )
    return model


def generate_digit_samples(
    vae_model: DigitVAE,
    num_samples: int = 12,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> List[Image.Image]:
    """
    Generates synthetic handwritten digits by sampling latent vectors z ~ N(0, (temperature)^2 * I).
    Returns a list of PIL Images (28x28 grayscale).
    """
    if device is None:
        device = get_device()

    vae_model.eval()
    vae_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        samples = vae_model.sample(
            num_samples=num_samples,
            device=device,
            temperature=temperature
        )  # (num_samples, 1, 28, 28)

    samples_np = samples.squeeze(1).cpu().numpy()  # (num_samples, 28, 28)
    # Clip and convert to uint8 [0, 255]
    samples_uint8 = np.clip(samples_np * 255.0, 0.0, 255.0).astype(np.uint8)

    pil_images = [Image.fromarray(img, mode="L") for img in samples_uint8]
    return pil_images


def reconstruct_digit(
    vae_model: DigitVAE,
    raw_tensor: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[Image.Image, Image.Image]:
    """
    Encodes an input digit into latent space and reconstructs it.
    Returns (original_pil, reconstructed_pil).
    """
    if device is None:
        device = get_device()

    vae_model.eval()
    vae_model.to(device)
    raw_tensor = raw_tensor.to(device)

    with torch.no_grad():
        recon_x, _, _ = vae_model(raw_tensor)

    orig_np = raw_tensor.squeeze().cpu().numpy()
    recon_np = recon_x.squeeze().cpu().numpy()

    orig_pil = Image.fromarray(np.clip(orig_np * 255.0, 0, 255).astype(np.uint8), mode="L")
    recon_pil = Image.fromarray(np.clip(recon_np * 255.0, 0, 255).astype(np.uint8), mode="L")

    return orig_pil, recon_pil


def interpolate_latents(
    vae_model: DigitVAE,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    steps: int = 8,
    device: Optional[torch.device] = None
) -> List[Image.Image]:
    """
    Interpolates linearly between the latent representations of two digits.
    Demonstrates smooth manifold geometry in the generative latent space.
    """
    if device is None:
        device = get_device()

    vae_model.eval()
    vae_model.to(device)

    with torch.no_grad():
        mu_a, _ = vae_model.encode(tensor_a.to(device))
        mu_b, _ = vae_model.encode(tensor_b.to(device))

        alphas = np.linspace(0.0, 1.0, steps)
        interpolated_imgs = []

        for alpha in alphas:
            z_interp = (1.0 - alpha) * mu_a + alpha * mu_b
            recon = vae_model.decode(z_interp)
            recon_np = recon.squeeze().cpu().numpy()
            img_pil = Image.fromarray(np.clip(recon_np * 255.0, 0, 255).astype(np.uint8), mode="L")
            interpolated_imgs.append(img_pil)

    return interpolated_imgs
