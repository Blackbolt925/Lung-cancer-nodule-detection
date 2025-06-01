# Enhanced Lung Nodule Detection with Hybrid CNN-Transformer Architecture

This repository presents an advanced deep learning pipeline for early-stage lung cancer detection using 3D CT scans. It implements a state-of-the-art hybrid CNN-Transformer model that significantly outperforms traditional 2D CNN approaches in both accuracy and robustness. The model is trained and evaluated on the LUNA-16 dataset, with specialized preprocessing and augmentation tailored for medical imaging.

## Key Features

- **Hybrid Architecture**  
  Combines 3D Convolutional Neural Networks (CNNs) for local spatial feature extraction with Transformers for global context modeling.

- **Advanced Image Enhancement**  
  Utilizes multi-scale wavelet decomposition and adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance CT scan quality and improve feature clarity.

- **Medical-Specific Augmentation**  
  Implements the MedMix strategy, designed to preserve anatomical correctness while augmenting training data.

- **3D Volumetric Analysis**  
  Processes entire 3D CT volumes rather than isolated 2D slices, enabling more accurate spatial understanding.

- **Self-Supervised Learning**  
  Applies contrastive learning to leverage unlabeled data for robust feature extraction.

- **Clinical Evaluation Focus**  
  Includes clinically meaningful evaluation metrics with a strong emphasis on reducing false negatives.

## Performance Summary

- **AUC-ROC**: Significantly higher than baseline 2D CNN approaches.
- **False Negative Reduction**: Prioritized in training and validation to support real-world cancer screening applications.
- **Cross-Dataset Validation**: Demonstrates robust generalization across different CT scanner types and imaging protocols.

## Technologies Used

- PyTorch Lightning  
- MONAI (Medical Open Network for AI)  
- SimpleITK  
- OpenCV  
- Scikit-learn  
- PyWavelets  
- NumPy and Pandas

## Dataset

- **LUNA-16 (Lung Nodule Analysis 2016)**  
  Comprehensive preprocessing pipeline includes:
  - Lung segmentation
  - HU normalization
  - Isotropic voxel resampling

## Clinical Impact

- Designed for deployment in real-world clinical workflows
- Emphasis on minimizing false negatives to enhance early cancer detection
- Generates uncertainty scores to assist radiologist review
- Preserves model interpretability for clinical decision-making

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for training)
- PyTorch 2.0+, MONAI, and supporting libraries


