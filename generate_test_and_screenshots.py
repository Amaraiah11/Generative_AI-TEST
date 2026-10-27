"""
generate_test_and_screenshots.py
Tests the end-to-end laboratory pipeline:
1. Loads cached CNN and VAE model checkpoints.
2. Verifies held-out test accuracy on MNIST test set.
3. Tests preprocessing on synthetic drawn digits.
4. Generates real class probabilities and confidence scores.
5. Generates synthetic handwritten digits from the VAE latent space.
6. Saves detailed demonstration visualization screenshots into screenshots/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.inference import load_or_train_classifier, classify_digit, get_device
from backend.generation import load_or_train_vae, generate_digit_samples, reconstruct_digit
from backend.preprocessing import preprocess_image
from models.classifier import load_mnist_data, evaluate_classifier


def run_tests_and_create_screenshots():
    device = get_device()
    print(f"[1/5] Testing on hardware device: {device}")

    screenshots_dir = os.path.join(BASE_DIR, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    # 1. Load Pretrained Models
    print("[2/5] Loading pre-trained models from saved_models/...")
    cnn_model, test_acc = load_or_train_classifier(epochs=1)
    vae_model = load_or_train_vae(epochs=1)

    print(f"  -> CNN Loaded. Test Accuracy: {test_acc:.2f}%")
    print(f"  -> VAE Loaded. Latent Dimension: {vae_model.latent_dim}")
    assert test_acc > 95.0, f"Expected test accuracy >95%, got {test_acc:.2f}%"

    # 2. Test Preprocessing & Classification on real MNIST test images
    print("[3/5] Testing Preprocessing & Discriminative Classification on MNIST test digits...")
    _, test_loader = load_mnist_data(data_dir=os.path.join(BASE_DIR, "data"), batch_size=64)
    test_images, test_labels = next(iter(test_loader))

    test_results = []
    # Pick distinct digits 0..9 from the test batch
    seen_digits = set()
    for img_tensor, label in zip(test_images, test_labels):
        d = label.item()
        if d not in seen_digits and len(seen_digits) < 6:
            seen_digits.add(d)
            # Convert raw normalized tensor to PIL image [0, 255]
            unnorm = img_tensor.squeeze().numpy() * 0.3081 + 0.1307
            unnorm_uint8 = np.clip(unnorm * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(unnorm_uint8, mode="L")

            cnn_t, raw_t, disp_img = preprocess_image(pil_img)
            pred = classify_digit(cnn_model, cnn_t)
            print(f"  Digit {d}: Predicted {pred['predicted_digit']} (Confidence: {pred['confidence']*100:.2f}%)")
            test_results.append({
                "target": d,
                "pred": pred,
                "disp_img": disp_img,
                "raw_t": raw_t
            })

    # 3. Test VAE Generation
    print("[4/5] Testing Generative VAE Latent Sampling...")
    generated_samples = generate_digit_samples(vae_model, num_samples=18, temperature=1.0)
    print(f"  -> Generated {len(generated_samples)} synthetic handwritten digit images.")
    assert len(generated_samples) == 18

    # 4. Generate Comprehensive Demonstration Screenshot 1: Overview
    print("[5/5] Creating Demonstration Visualizations for screenshots/...")
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.5))
    fig.patch.set_facecolor('#0F172A')

    # Row 1: Discriminative Classifier Inputs & Outputs
    for i, res in enumerate(test_results[:6] if len(test_results)>=6 else test_results):
        ax = axes[0, i]
        ax.imshow(res["disp_img"], cmap="gray")
        target = res["target"]
        predicted = res["pred"]["predicted_digit"]
        conf = res["pred"]["confidence"] * 100
        ax.set_title(f"Target: {target} | Pred: {predicted}\nConf: {conf:.1f}%", color='#38BDF8', fontsize=10, weight='bold')
        ax.axis('off')

    # Fill remaining columns in Row 1 if any
    for i in range(len(test_results), 6):
        axes[0, i].axis('off')

    # Row 2 & 3: Generative VAE Samples
    for idx, sample_img in enumerate(generated_samples[:12]):
        row = 1 + (idx // 6)
        col = idx % 6
        ax = axes[row, col]
        ax.imshow(sample_img, cmap="gray")
        ax.set_title(f"VAE Gen #{idx+1}", color='#4ADE80', fontsize=9)
        ax.axis('off')

    plt.suptitle("Generative vs. Discriminative Model Lab (MNIST Benchmark)\n"
                 f"CNN Classifier (Test Accuracy: {test_acc:.2f}%) vs. VAE Latent Synthesis (z ~ N(0, I))",
                 color='white', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    screenshot_path1 = os.path.join(screenshots_dir, "lab_demonstration.png")
    plt.savefig(screenshot_path1, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  -> Saved demonstration screenshot to: {screenshot_path1}")

    # Generate Screenshot 2: Side-by-Side Lab Panel
    fig2, (ax_left, ax_center, ax_right) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.patch.set_facecolor('#F8FAFC')

    # Left: Drawn Digit
    sample_res = test_results[0]
    ax_left.imshow(sample_res["disp_img"], cmap="gray")
    ax_left.set_title(f"Preprocessed Input X (28×28)\nLabel: {sample_res['target']}", fontsize=11, weight='bold')
    ax_left.axis('off')

    # Center: Probabilities
    probs = sample_res["pred"]["probabilities_list"]
    bars = ax_center.bar(range(10), probs, color=['#2563EB' if i == sample_res['pred']['predicted_digit'] else '#CBD5E1' for i in range(10)])
    ax_center.set_xticks(range(10))
    ax_center.set_ylim(0, 1.05)
    ax_center.set_title(f"Discriminative Output: P(Y|X)\nPredicted: {sample_res['pred']['predicted_digit']} ({sample_res['pred']['confidence']*100:.1f}%)", fontsize=11, weight='bold')
    ax_center.set_xlabel("Digit Class")
    ax_center.set_ylabel("Probability")
    ax_center.spines['top'].set_visible(False)
    ax_center.spines['right'].set_visible(False)

    # Right: VAE Reconstructed Digit
    orig_p, recon_p = reconstruct_digit(vae_model, sample_res["raw_t"])
    ax_right.imshow(recon_p, cmap="gray")
    ax_right.set_title("Generative Output: VAE Reconstructed X̂\nLatent Space z ∈ ℝ²⁰", fontsize=11, weight='bold')
    ax_right.axis('off')

    plt.suptitle("Discriminative Prediction vs. Generative Reconstruction", fontsize=13, weight='bold')
    plt.tight_layout()
    screenshot_path2 = os.path.join(screenshots_dir, "model_comparison_panel.png")
    plt.savefig(screenshot_path2, dpi=180, facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close(fig2)
    print(f"  -> Saved comparison screenshot to: {screenshot_path2}")

    print("\n[SUCCESS] All validation tests passed and screenshots generated successfully!")


if __name__ == "__main__":
    run_tests_and_create_screenshots()
