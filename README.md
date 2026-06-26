# DeepFake Detection using ResNet50 & Transfer Learning

An end-to-end Deep Learning pipeline developed to classify real photographs from AI-generated synthetic images using the balanced **CIFAKE dataset** (100,000 RGB images).

## 🚀 Project Overview
With the rapid rise of Generative AI tools (Stable Diffusion, Midjourney), distinguishing synthetic media from reality has become nearly impossible for the human eye. This project implements a robust Transfer Learning baseline using a pre-trained **ResNet50** architecture to automate media verification and combat disinformation.

## 📊 Dataset Configuration
- **Dataset Used:** CIFAKE (derived from CIFAR-10 and Stable Diffusion 1.4)
- **Total Images:** 100,000 RGB colored images (32x32 resolution)
- **Data Split:**
  - **Training/Validation:** 80,000 images (80% Train / 20% Val)
  - **Testing:** 20,000 unseen images (10,000 Real / 10,000 Fake)

## 🏗️ Architecture & Implementation
The core framework leverages **Transfer Learning** to evaluate how effectively high-resolution pre-trained networks extract low-resolution spatial features:
1. **Base Model:** ResNet50 (Pre-trained on ImageNet) with weights **Frozen** (`trainable=False`) to avoid computational overhead.
2. **Global Pooling:** `GlobalAveragePooling2D()` to flatten multi-dimensional CNN outputs.
3. **Dense Layer:** 512 Hidden units with **ReLU** activation.
4. **Regularization:** **50% Dropout** to mitigate overfitting.
5. **Output Layer:** Single neural unit with **Sigmoid** activation for binary classification.

## 📈 Final Evaluation Metrics
The model achieved highly stable convergence and excellent generalization across 10 epochs:
- **Validation Accuracy:** ~79.65%
- **Validation Loss:** ~0.4402
- **Precision (Fake Class):** 0.79
- **Recall (Fake Class):** 0.79
- **F1-Score:** 0.79

### Key Insights (The Background Bias)
Based on benchmark research, deep fake image detection often focuses heavily on high-frequency pixel artifacts and compression grid lines left in the image backgrounds rather than central semantic objects (e.g., cats, cars). The frozen ResNet50 feature extractor successfully captures these high-frequency background grids to make accurate boundary predictions.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Frameworks:** TensorFlow / Keras
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab / Jupyter Notebook

## 👥 Project Contributor
- **Abdullah Shamas** 
