# SEGAN: Semantic-Enhanced GAN for GNSS Anomaly Detection in Offshore Pile Driving

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"SEGAN: A Semantic-Enhanced Generative Adversarial Network for GNSS Anomaly Detection and Reconstruction in Offshore Pile Driving Operations"**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This repository provides the complete implementation of **SEGAN**, a novel deep learning framework designed for real-time GNSS anomaly detection and reconstruction in offshore pile driving construction. The framework addresses critical challenges in maintaining precise pile positioning under dynamic offshore environments by effectively identifying and correcting missing, constant, and mutation-type anomalies.

**Key Contributions:**
- **Temporal Sampling Layer (TSL)**: Captures local temporal dependencies and reduces interference from preceding anomalies in time-series data.
- **Eliminator Mechanism**: Enhances the discriminator to differentiate between normal and anomalous patterns more effectively.
- **Real-Time Performance**: Achieves 6.8 ms inference time, well within the 500 ms GNSS refresh rate.

---

## ✨ Key Features

- ✅ **State-of-the-art Reconstruction Accuracy**: Outperforms AE, VAE, and TadGAN models in GNSS coordinate reconstruction.
- ✅ **Robust Anomaly Detection**: Achieves superior F1 score and MCC values across multiple test datasets.
- ✅ **Ablation-Tested Design**: Comprehensive evaluation of TSL contribution and threshold sensitivity.
- ✅ **Full Reproducibility**: Complete hyperparameter configurations and training details provided.
- ✅ **Multiple Baseline Implementations**: Includes AE, VAE, and TadGAN for direct comparison.

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 1.10+

### Required Packages

```txt
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
tqdm>=4.62.0
```

---

## 🏗 Model Architecture

### SEGAN Framework

The SEGAN model consists of four main components:

1. **Generator (G)**: Reconstructs GNSS time-series data using encoder-decoder architecture with TSL.
2. **Discriminator (D)**: Distinguishes between real and reconstructed sequences.
3. **Temporal Sampling Layer (TSL)**: Extracts high-level temporal features with configurable window size and stride.
4. **Eliminator Mechanism**: Enhances discriminator's ability to classify anomalous patterns.



---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📌 Notes

- **Data Availability**: Due to confidentiality agreements with construction partners, the full raw GNSS dataset cannot be publicly released.
- **Hardware Requirements**: Training SEGAN on the full dataset requires approximately 16 GB GPU memory. For lower memory configurations, reduce batch size or sequence length.
- **Inference Time**: The reported 6.8 ms inference time was measured on an NVIDIA RTX 3090 GPU. Performance may vary on different hardware.

---

**Last Updated**: April 2026

如果您需要调整某些部分或添加其他内容，请随时告诉我！
