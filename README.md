# Generative vs. Discriminative Model Lab (MNIST)

A local web application contrasting **Discriminative Modeling** (Convolutional Neural Network) against **Generative Modeling** (Variational Autoencoder) using PyTorch, Flask, and the MNIST handwritten digit benchmark. Runs on **localhost** (`http://127.0.0.1:5000`) and is designed for easy local execution and manual deployment.

---

## 📌 Project Overview

This laboratory compares the two fundamental paradigms in machine learning:
1. **Discriminative Modeling ($P(Y \mid X)$)**: Deep **Convolutional Neural Network (CNN)** predicting class label $Y \in \{0, \dots, 9\}$ given an input digit image $X$.
2. **Generative Modeling ($P(X)$ or $P(X \mid Z)P(Z)$)**: **Variational Autoencoder (VAE)** modeling continuous latent space $z \in \mathbb{R}^{20}$ to synthesize novel handwritten digits from Gaussian noise.

Features:
- Interactive HTML5 drawing canvas to draw digits with your mouse/touch.
- Image uploader (PNG, JPG) and pre-loaded authentic MNIST test sample loader.
- Real-time digit classification with confidence score and 0–9 probability bar chart.
- Held-out test accuracy calculation on 10,000 independent MNIST test digits (**98.91%**).
- Dynamic sample slider (1 to 20) and VAE latent space sample generator.
- Side-by-side conceptual comparison table and theoretical explanation.

---

## ⚖️ Generative vs. Discriminative Models

| Dimension | Discriminative Model (CNN) | Generative Model (VAE) |
| :--- | :--- | :--- |
| **Primary Goal** | Separate classes by finding optimal decision boundaries | Learn the underlying probability distribution of the data |
| **Mathematical Formulation** | Estimates conditional posterior **$P(Y \mid X)$** | Estimates marginal/joint likelihood **$P(X) = \int P(X \mid z) P(z) dz$** |
| **Input** | High-dimensional image pixels $X \in \mathbb{R}^{1 \times 28 \times 28}$ | Latent Gaussian noise vector $z \sim \mathcal{N}(0, I)$ |
| **Output** | Discrete categorical probability distribution over classes 0–9 | Complete synthetic $28 \times 28$ image matrix with pixel values in $[0, 1]$ |
| **Training Objective** | Minimize Cross-Entropy Loss: $\mathcal{L}_{\text{CE}} = - \sum_{k=0}^{9} y_k \log(\hat{y}_k)$ | Maximize Evidence Lower Bound (ELBO): $\mathcal{L}_{\text{VAE}} = \text{BCE}(X, \hat{X}) + D_{\text{KL}}(q(z \mid X) \parallel p(z))$ |
| **Can It Generate New Images?** | **No.** Discards intra-class variation; has no mechanism to sample pixels | **Yes.** Can synthesize an infinite variety of plausible handwritten digits |
| **Latent Space** | Task-specific discriminative feature maps (no prior distribution) | Structured, continuous Gaussian manifold allowing smooth interpolation |
| **Primary Use Cases** | Classification, object detection, segmentation | Data augmentation, anomaly detection, creative synthesis, denoising |

---

## 📁 Project Structure

```text
generative-discriminative-lab/
├── app.py                     # Flask web server & API endpoints
├── templates/
│   └── index.html             # Responsive HTML5 Canvas frontend
├── requirements.txt           # Core project dependencies
├── README.md                  # Documentation and deployment guide
├── train_all.py               # Pre-training and checkpoint verification script
├── digits_100.csv             # Sample digit dataset reference
├── models/
│   ├── __init__.py
│   ├── classifier.py          # PyTorch CNN architecture & evaluation
│   └── generator.py           # PyTorch VAE architecture & latent sampling
├── backend/
│   ├── __init__.py
│   ├── preprocessing.py       # Grayscale, bounding box centering & normalization
│   ├── inference.py           # Classifier loader, device management, prediction
│   └── generation.py          # VAE loader, latent space sampling
├── saved_models/
│   ├── cnn_classifier.pth     # Saved weights for CNN classifier (98.91% test acc)
│   └── vae_generator.pth      # Saved weights for VAE generator
├── data/                      # MNIST dataset cache directory
└── screenshots/               # Demonstration captures
```

---

## 🛠️ Technology Stack

- **Backend:** Python 3.10+, Flask
- **Deep Learning Framework:** PyTorch & Torchvision
- **Frontend:** HTML5 Canvas, Vanilla JavaScript, CSS3 (zero external frontend frameworks)
- **Image Processing:** Pillow (PIL), NumPy

---

## 🚀 How to Run Locally

### 1. Open Terminal in VS Code
```powershell
cd generative-discriminative-lab
```

### 2. Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application
```powershell
python app.py
```

### 4. Open in Your Browser
Open: **[http://localhost:5000](http://localhost:5000)** (or `http://127.0.0.1:5000`).

---

## 🌐 Manual Deployment Guide

Because this is a standard Flask web application (without Streamlit), you can deploy it manually anywhere:

### Option A: Deploy with Gunicorn / WSGI (Linux / Cloud VPS / Render)
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:$PORT app:app
```

### Option B: Deploy with Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```
Build and run:
```bash
docker build -t mnist-lab .
docker run -p 5000:5000 mnist-lab
```

---

## 📄 License
Educational and research laboratory implementation.
