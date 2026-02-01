# ForgeryNet-CNN-Based-Image-Forgery-Detection-and-Localization-
ForgeryNet is an AI-based system developed to detect whether a digital image  has been manipulated and to visually highlight the forged regions.

## Overview
This project explores convolutional neural network (CNN)–based methods for detecting and localizing forged regions in digitally manipulated images. The primary objective is to distinguish between real and tampered images and provide visual explanations of manipulated regions using explainability techniques.

## Motivation
With the widespread availability of image editing tools, digital image manipulation has become increasingly common. Traditional object detection methods often fail to capture subtle forgery patterns, as manipulated regions do not always exhibit clear shapes or boundaries. This project was motivated by the need to understand how deep learning models can identify texture-level inconsistencies and provide interpretable results for forgery detection.

## Approach
The project involved experimenting with multiple deep learning architectures to identify the most suitable model for forgery detection:

Initial exploration with object detection and segmentation-based approaches
Transition to CNN-based classification models due to the texture-based nature of forgery
Use of EfficientNet-B3 (pre-trained on ImageNet) for improved feature extraction
Application of Grad-CAM to visualize and interpret model predictions

## Dataset
The model was trained and evaluated using a  CASIA v2 dataset containing copy-move and splicing manipulations. Dataset files are not included in this repository.

## Tools and Technologies
Programming Language: Python 
Deep Learning Framework: PyTorch 
Models: YOLO, YOLOv5, RDS-YOLO (experiments), EfficientNet-B3 
(final model) 
Explainability Tool: Grad-CAM 
Libraries: OpenCV, NumPy, Matplotlib 
Dataset: CASIA v2 (Real and Tampered Images + Ground Truth Masks) 
IDE / Environment: Google Colab, VS Code 
UI Framework: Streamlit for dashboard development 

## Experiments & Observations
Object detection models were not effective due to the absence of distinct object boundaries in forged regions
CNN-based models demonstrated stronger performance in capturing texture inconsistencies
Explainability techniques helped validate model focus on manipulated regions
A Streamlit dashboard was built to allow users to upload images and 
view four outputs side-by-side: 
• Original Image 
• Grad-CAM Heatmap 
• Predicted Tampered Outline 
• Ground Truth Mask (for fake images) 

## Limitations & Future Work
This project is a research-oriented prototype and has several limitations:
Limited generalization to AI-generated image forgeries
Sensitivity to dataset distribution and manipulation types
Future work includes extending the approach to deepfake and AI-generated image detection, as well as improving robustness through larger datasets.

## Disclaimer
This repository is intended for academic and learning purposes only. It does not represent a production-ready system.

