"""
app.py
Generative vs. Discriminative Model Lab (Flask Localhost Application)
Runs locally on localhost (http://127.0.0.1:5000) with PyTorch CNN & VAE models.
Ready for local execution and manual deployment.
"""

import os
import sys
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.preprocessing import preprocess_image
from backend.inference import (
    load_or_train_classifier,
    classify_digit,
    get_device,
    DEFAULT_CLASSIFIER_PATH
)
from backend.generation import (
    load_or_train_vae,
    generate_digit_samples,
    DEFAULT_VAE_PATH
)
from models.classifier import load_mnist_data

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load pre-trained models into memory
device = get_device()
print(f"[INFO] Initializing PyTorch models on {device}...")
cnn_model, test_accuracy = load_or_train_classifier(model_path=DEFAULT_CLASSIFIER_PATH, epochs=3)
vae_model = load_or_train_vae(model_path=DEFAULT_VAE_PATH, epochs=4)
print(f"[INFO] Models loaded successfully! Held-out Test Accuracy: {test_accuracy:.2f}%")


def pil_to_base64(img: Image.Image) -> str:
    """Helper to convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    """Serves the main laboratory web UI."""
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def info():
    """Returns system device and test accuracy."""
    return jsonify({
        "device": str(device).upper(),
        "test_accuracy": round(test_accuracy, 2)
    })


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Discriminative inference endpoint:
    Accepts base64 image data, preprocesses to 28x28, runs CNN classification,
    and returns predictions and probabilities.
    """
    data = request.get_json()
    if not data or "image_data" not in data:
        return jsonify({"error": "No image data provided"}), 400

    data_url = data["image_data"]
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(data_url)
        pil_img = Image.open(io.BytesIO(raw_bytes))

        # Preprocess using backend pipeline
        cnn_tensor, _, disp_img = preprocess_image(pil_img)
        res = classify_digit(cnn_model, cnn_tensor)

        return jsonify({
            "predicted_digit": res["predicted_digit"],
            "confidence": res["confidence"],
            "probabilities_list": res["probabilities_list"],
            "preprocessed_image": pil_to_base64(disp_img),
            "test_accuracy": test_accuracy
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generative inference endpoint:
    Samples VAE continuous latent space z ~ N(0, I) and decodes into synthetic digit images.
    """
    data = request.get_json() or {}
    count = int(data.get("count", 10))
    count = max(1, min(24, count))

    try:
        samples = generate_digit_samples(vae_model, num_samples=count)
        samples_b64 = [pil_to_base64(img) for img in samples]
        return jsonify({"samples": samples_b64})
    except Exception as e:
        return jsonify({"error": f"Generation error: {str(e)}"}), 500


@app.route("/api/sample", methods=["GET"])
def get_sample():
    """Returns an authentic test sample from the MNIST test set."""
    try:
        _, test_loader = load_mnist_data(batch_size=32)
        imgs, labels = next(iter(test_loader))
        idx = np.random.randint(0, len(imgs))
        unnorm = (imgs[idx].squeeze().numpy() * 0.3081 + 0.1307) * 255.0
        pil_img = Image.fromarray(np.clip(unnorm, 0, 255).astype(np.uint8), mode="L")
        return jsonify({
            "image": pil_to_base64(pil_img),
            "label": int(labels[idx].item())
        })
    except Exception as e:
        return jsonify({"error": f"Could not fetch sample: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nRunning locally on: http://localhost:{port} (http://127.0.0.1:{port})")
    print("Press Ctrl+C to stop the server.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
