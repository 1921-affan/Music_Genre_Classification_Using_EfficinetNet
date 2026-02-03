# 👁️ Eye Disease Classification using Deep Learning

A high-performance AI system for detecting retinal diseases from fundus images, featuring advanced preprocessing and a fine-tuned Xception model (88.85% Accuracy).

## 🚀 Key Features

### 1. Advanced Preprocessing (Stage 1)
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for vessel enhancement.
- **Ben Graham's Method** for automatic cropping and color normalization.
- Removes black margins and standardizes lighting conditions.

### 2. Champion Model Architecture (Stage 2)
- **Base**: Xception (Pre-trained on ImageNet).
- **Strategy**: Phase 4 "Clean Rebuild" with Weight Transfer.
- **Accuracy**: **88.85%** (Surpassing standard benchmarks).
- **Optimization**: Tuned using aggressive regularization (L2 + Dropout) to prevent overfitting.

### 3. Deployment (Stage 4)
- **Interface**: Custom Streamlit Web App.
- **Inference**: Real-time classification on local CPU/GPU.
- **Stability**: Fixed "Double Normalization" bugs for robust production usage.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1921-affan/Music_Genre_Classification_Using_EfficinetNet.git
   cd Music_Genre_Classification_Using_EfficinetNet
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model:**
   *Note: The Xception model (`xception_final_boost.h5`) is stored using Git LFS due to its size (206MB).*
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

## 📊 Performance Metrics

| Model | Accuracy | Status |
|-------|----------|--------|
| **Xception** | **88.85%** | 🏆 **Deployed** |
| Ensemble (Xception+EffNet) | 87.78% | Deprecated (Complexity) |
| EfficientNetB3 | 83.60% | Archived |
| MobileNetV2 | 79.00% | Baseline |

---
*Project for ANN4142 | 2026*
