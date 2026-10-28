<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=581C87,7E22CE,C084FC&height=220&section=header&text=Generative%20vs%20Discriminative%20Lab&fontSize=40&fontColor=ffffff&fontAlignY=38&desc=Interactive%20Streamlit%20MNIST%20Synthesis%20%26%20CNN%20Classification&descSize=16&descAlignY=60&descAlign=50" width="100%" alt="Generative vs Discriminative Lab Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Backend-PyTorch%20%7C%20Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Paradigm-Generative%20vs%20Discriminative-7E22CE?style=for-the-badge" alt="Generative vs Discriminative" />
  <img src="https://img.shields.io/badge/Status-Fully%20Deployed-success?style=for-the-badge" alt="Status" />
</p>

<a id="overview"></a>
## 📖 Executive Summary & Overview

Machine learning models divide fundamentally into **Discriminative** approaches (modeling $P(Y|X)$, estimating decision boundaries) and **Generative** approaches (modeling $P(X)$ or $P(X,Y)$, capturing underlying data distributions). Visualizing this theoretical contrast in a live, interactive environment clarifies model behavior for students and engineers.

The **Generative vs Discriminative Model Lab** is a full-featured web application engineered with **Streamlit** and **PyTorch**. The application enables users to generate synthetic handwritten digits across latent manifolds while evaluating how a fine-tuned Convolutional Neural Network (CNN) perceives, classifies, and assigns confidence bounds to both synthetic and handwritten inputs.

<a id="toc"></a>
## 📑 Table of Contents

- [📖 Executive Summary & Overview](#overview)
- [🎯 Key Features & Capabilities](#features)
- [🏗️ System Architecture & Pipeline](#architecture)
- [🛠️ Tech Stack Breakdown](#tech-stack)
- [⚙️ Reproducible Quickstart & Setup](#setup)
- [📁 Project Directory Layout](#project-layout)
- [👨‍💻 Author & Contributions](#author)

<a id="features"></a>
## 🎯 Key Features & Capabilities

- **Interactive Streamlit Web Dashboard:** Clean, responsive UI with real-time slider controls for latent vector tuning.
- **High-Accuracy CNN Classifier:** Convolutional neural network trained with data augmentation achieving >99% test accuracy on MNIST.
- **Generative Synthesis Engine:** Real-time generation of custom handwritten digits from latent seeds.
- **Classification Confidence Visualizer:** Softmax probability bar charts detailing class distributions for synthetic images.
- **Automated Verification Harness:** `test_handwriting.py` and `generate_test_and_screenshots.py` for headless automated regression testing.

<a id="architecture"></a>
## 🏗️ System Architecture & Pipeline

```mermaid
flowchart LR
    subgraph UI["Streamlit Frontend"]
        Controls["Latent Dimension Controls"]
        Canvas["Interactive Canvas / Sample Picker"]
    end

    subgraph Backend["PyTorch Backend Modules"]
        GenEngine["backend/generation.py<br/>(Generative Model)"]
        Classifier["finetune_cnn.py<br/>(CNN Discriminator)"]
    end

    subgraph Output["Real-Time Inference Output"]
        ImgOut["Rendered 28x28 Image"]
        BarChart["Class Confidence Distribution"]
    end

    Controls --> GenEngine
    Canvas --> Classifier
    GenEngine --> ImgOut
    ImgOut --> Classifier
    Classifier --> BarChart
```

<a id="tech-stack"></a>
## 🛠️ Tech Stack Breakdown

| Component | Library / Framework | Function |
| :--- | :--- | :--- |
| **Frontend Framework** | Streamlit | Reactive web dashboard, sliders, and image rendering |
| **Deep Learning Engine** | PyTorch / Torchvision | Tensor operations, CNN architecture, and inference |
| **Data Processing** | NumPy, Pandas, Pillow (PIL) | Pixel array manipulation, normalization, image export |
| **Automation & Testing** | Pytest, Custom Test Harnesses | Headless UI validation and model convergence tests |

<a id="setup"></a>
## ⚙️ Reproducible Quickstart & Setup

### Prerequisites
- Python 3.9+

```bash
# 1. Clone repository
git clone https://github.com/Amaraiah11/Generative_AI-TEST.git
cd Generative_AI-TEST

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch web application
streamlit run app.py
```

<a id="project-layout"></a>
## 📁 Project Directory Layout

```text
Generative_AI-TEST/
├── app.py                             # Main Streamlit web application
├── finetune_cnn.py                    # High-accuracy CNN classifier fine-tuning script
├── generate_test_and_screenshots.py  # Automated visual regression test harness
├── test_handwriting.py                # Unit and integration test suite
├── train_all.py                       # Comprehensive training coordinator
├── digits_100.csv                     # Sample benchmark handwritten digits dataset
├── backend/
│   └── generation.py                  # Generative model architecture & inference logic
├── requirements.txt                   # Production dependencies
└── README.md                          # Interactive lab documentation
```

<a id="author"></a>
## 👨‍💻 Author & Contributions

This project is developed and maintained with high technical rigour by **Amaraiah C**, Software Engineer and Data Analytics Practitioner studying Bachelor of Computer Applications (BCA) at Alliance University.

- 🌐 **GitHub:** [@Amaraiah11](https://github.com/Amaraiah11)
- 💼 **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/amaraiah-c/)
- 📧 **Institutional Contact:** `camaraiahBCA24@ced.alliance.edu.in`

### 🤝 Contribution Guidelines
Contributions, bug reports, and architectural discussions are welcome!
1. **Fork** the repository: `https://github.com/Amaraiah11/Generative_AI-TEST`
2. **Branch**: `git checkout -b feature/your-feature-name`
3. **Commit**: `git commit -m "feat: enhance feature logic"`
4. **Push**: `git push origin feature/your-feature-name`
5. **Open Pull Request** against the `Amaraiah-C` branch.

---

<p align="center">
  <b>⭐ If this codebase or documentation assisted your engineering workflows, please star the repository! ⭐</b>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=581C87,7E22CE,C084FC&height=100&section=footer" width="100%" alt="Footer Banner" />
</p>
