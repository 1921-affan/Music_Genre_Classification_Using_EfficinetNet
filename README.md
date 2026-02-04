# 👁️ Eye Disease Classification using Deep Learning

A high-performance AI system for detecting retinal diseases from fundus images, featuring advanced preprocessing and a fine-tuned Xception model (**88.85% Accuracy**).

## 🚀 Key Features

### 1. Dataset 🧵
- **Source**: [Kaggle - Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Content**: 4,217 images across 4 categories (Normal, Cataract, Glaucoma, Diabetic Retinopathy).
- **Structure**: 4 subdirectories, ~1050 images per class.

### 2. Advanced Preprocessing
- **Base**: Xception (Pre-trained on ImageNet).
- **Performance**: **88.85% Accuracy** on the test set.
- **Optimization**: Tuned using aggressive regularization (L2 + Dropout) to prevent overfitting and ensure robust generalization.

### 3. Explainable AI 🧠
- **Technique**: Grad-CAM (Gradient-weighted Class Activation Mapping).
- **Function**: Generates physiological heatmaps overlaid on patient scans.
- **Utility**: Highlights specific regions (e.g., optic nerve, lesions) driving the diagnosis, offering clinical interpretability.

### 4. Deployment Pipeline
- **Interface**: Custom Streamlit Web App.
- **Inference**: Real-time classification on local CPU/GPU.
- **Robustness**: Production-ready image enhancement pipeline matching training conditions.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1921-affan/Music_Genre_Classification_Using_EfficinetNet.git
   cd "Eye Disease Classification using DL/Model"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model:**
   *Note: The Xception model (`xception_final_boost.h5`) is stored using Git LFS due to its size.*
   ```bash
   git lfs pull
   ```

## 🖥️ Usage

Run the Streamlit application locally:

```bash
python -m streamlit run app.py
```

Upload a retinal fundus image (JPG/PNG) to get an instant diagnosis for:
*   Cataract
*   Diabetic Retinopathy
*   Glaucoma
*   Normal

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **88.85%** |
| Model Type | Xception (Deep CNN) |
| Feature | Class-Weighted Loss Handling |

---
*Project for ANN4142 | 2026*
