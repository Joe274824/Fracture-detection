# Fracture Detection using 3D CT Images

This project focuses on automatic detection of fractures from 3D CT scans using deep learning. It was developed as part of a research project at the University of Adelaide.

## 📌 Project Overview

- **Task**: Binary classification — Fractured vs. Healthy
- **Model**: 3D DenseNet
- **Data**: CT scan volumes in NIfTI format (`.nii`)
- **Goal**: Assist radiologists with accurate and efficient fracture identification

## 🧠 Key Features

- 3D DenseNet-based classifier
- Mixed precision training with GPU acceleration
- Support for large-scale NIfTI image processing
- Grad-CAM visualization for interpretability
- External validation with unseen data
