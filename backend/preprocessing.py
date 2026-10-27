"""
backend/preprocessing.py
Robust MNIST preprocessing engine for hand-drawn canvas strokes and uploaded image files.
Employs polarity detection, noise gating, stroke thickness normalization, aspect-ratio scaling to 20x20,
and Yann LeCun exact sub-pixel Center-of-Mass centering in 28x28 field.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch


def preprocess_image(
    input_data: Union[Image.Image, np.ndarray, bytes],
    target_size: Tuple[int, int] = (28, 28),
    min_active_pixels: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """
    Transforms any user-drawn canvas stroke or uploaded digit image into
    an authentic 28x28 MNIST-format tensor.

    Steps:
      1. Converts to grayscale.
      2. Detects background polarity and inverts if needed (ensuring white digit on black background).
      3. Gating & background noise suppression.
      4. Bounding-box detection with aspect ratio preservation (scaled to 20x20 box).
      5. Anti-aliasing with Lanczos filter and stroke thickness normalization.
      6. Yann LeCun center-of-mass alignment (sub-pixel affine shift to (13.5, 13.5)).
      7. Normalization: mean=0.1307, std=0.3081 for CNN.
    """
    if input_data is None:
        raise ValueError("No input image was provided. Please draw or upload an image.")

    # 1. Convert input to PIL Image in grayscale ('L')
    if isinstance(input_data, np.ndarray):
        if input_data.size == 0:
            raise ValueError("Canvas array is empty.")
        if input_data.ndim == 3 and input_data.shape[2] == 4:
            # 4-channel RGBA from canvas or transparent PNG
            alpha = input_data[:, :, 3]
            rgb = input_data[:, :, :3]
            if np.max(alpha) > 10 and np.min(alpha) < 250:
                pil_raw = Image.fromarray(input_data.astype(np.uint8), mode="RGBA")
                bg = Image.new("RGB", pil_raw.size, (255, 255, 255))
                bg.paste(pil_raw, mask=pil_raw.split()[3])
                pil_img = bg.convert("L")
            else:
                pil_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("L")
        elif input_data.ndim == 3:
            pil_img = Image.fromarray(input_data.astype(np.uint8)).convert("L")
        elif input_data.ndim == 2:
            pil_img = Image.fromarray(input_data.astype(np.uint8), mode="L")
        else:
            raise ValueError(f"Unsupported image dimensions: {input_data.shape}")
    elif isinstance(input_data, Image.Image):
        pil_img = input_data.convert("L")
    else:
        try:
            import io
            pil_img = Image.open(io.BytesIO(input_data) if isinstance(input_data, bytes) else input_data).convert("L")
        except Exception as e:
            raise ValueError(f"Could not open image file: {str(e)}")

    img_arr = np.array(pil_img, dtype=np.float32)

    # 2. Check for blank canvas
    if img_arr.std() < 1.0:
        raise ValueError("Canvas is blank. Please draw a clear digit.")

    # 3. Detect Polarity (MNIST = bright stroke on dark background)
    h, w = img_arr.shape
    corner_size = max(3, min(h, w) // 10)
    corners = [
        img_arr[:corner_size, :corner_size],
        img_arr[:corner_size, -corner_size:],
        img_arr[-corner_size:, :corner_size],
        img_arr[-corner_size:, -corner_size:]
    ]
    corner_mean = np.mean([c.mean() for c in corners])

    # If background is light (white paper / canvas), invert
    if corner_mean > 120.0:
        img_arr = 255.0 - img_arr

    # Enhance contrast: subtract lowest 10th percentile and rescale
    low_val = np.percentile(img_arr, 10)
    img_arr = np.clip((img_arr - low_val) * (255.0 / max(1.0, 255.0 - low_val)), 0.0, 255.0)

    # Suppress background noise
    img_arr[img_arr < 35.0] = 0.0

    # 4. Active bounding box
    active_y, active_x = np.nonzero(img_arr > 30.0)
    if len(active_y) < min_active_pixels:
        raise ValueError("No distinct digit strokes detected. Please draw a visible digit.")

    # Add small padding margin to bounding box
    pad = max(2, int(min(h, w) * 0.02))
    ymin = max(0, active_y.min() - pad)
    ymax = min(h, active_y.max() + pad + 1)
    xmin = max(0, active_x.min() - pad)
    xmax = min(w, active_x.max() + pad + 1)

    digit_crop = img_arr[ymin:ymax, xmin:xmax]
    crop_h, crop_w = digit_crop.shape

    # 5. Fit digit inside a 20x20 box preserving aspect ratio (LeCun standard)
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20.0)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20.0)))

    crop_pil = Image.fromarray(digit_crop.astype(np.uint8), mode="L")

    # If stroke is thin (e.g. quick pen trace), gently dilate with MaxFilter
    # Calculate density: ratio of active pixels to bounding box area
    density = len(active_y) / float(crop_h * crop_w)
    if density < 0.16:
        crop_pil = crop_pil.filter(ImageFilter.MaxFilter(3))

    resized_pil = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Place inside 28x28 canvas and center by center of mass
    canvas_28 = Image.new("L", target_size, color=0)
    canvas_28.paste(resized_pil, ((28 - new_w) // 2, (28 - new_h) // 2))

    arr28 = np.array(canvas_28, dtype=np.float32)
    total_mass = arr28.sum()

    if total_mass > 0:
        y_grid, x_grid = np.indices((28, 28))
        cy = float((y_grid * arr28).sum() / total_mass)
        cx = float((x_grid * arr28).sum() / total_mass)

        # Center of mass should be at (13.5, 13.5)
        dx = 13.5 - cx
        dy = 13.5 - cy

        # Clamp shift to prevent shifting outside frame
        dx = max(-5.0, min(5.0, dx))
        dy = max(-5.0, min(5.0, dy))

        # Sub-pixel smooth affine transformation
        shifted_pil = canvas_28.transform(
            target_size,
            Image.Transform.AFFINE,
            (1.0, 0.0, -dx, 0.0, 1.0, -dy),
            resample=Image.Resampling.BILINEAR
        )
    else:
        shifted_pil = canvas_28

    final_arr = np.array(shifted_pil, dtype=np.float32)

    # 7. Normalize tensors
    raw_norm = np.clip(final_arr / 255.0, 0.0, 1.0)
    raw_tensor = torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).float()

    # Standard MNIST Normalization for CNN
    cnn_norm = (raw_norm - 0.1307) / 0.3081
    cnn_tensor = torch.from_numpy(cnn_norm).unsqueeze(0).unsqueeze(0).float()

    display_image = Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8), mode="L")

    return cnn_tensor, raw_tensor, display_image
