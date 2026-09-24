# Medical Image Classification

## Project Overview

This project classifies chest X-ray images into NORMAL and PNEUMONIA categories using deep learning and transfer learning.

## Dataset

Chest X-Ray Pneumonia Dataset.

Classes:

- NORMAL
- PNEUMONIA

## Model

MobileNetV2 with ImageNet pretrained weights.

Transfer learning is used to reduce training time and improve performance.

## Training

- Image Size: 224 x 224
- Batch Size: 32
- Epochs: Up to 10
- Optimizer: Adam
- Learning Rate: 0.0001
- GPU: Google Colab GPU when available

## Actual Results

| Metric | Score |
|---|---:|
| Accuracy | 0.7772 |
| Precision | 0.7418 |
| Recall | 0.9872 |
| F1 Score | 0.8471 |

## Workflow

Chest X-ray
↓
Image Preprocessing
↓
Data Augmentation
↓
MobileNetV2
↓
Feature Extraction
↓
Classification
↓
NORMAL / PNEUMONIA

## FastAPI Deployment

The trained MobileNetV2 model is integrated with FastAPI.

### GET /

Checks API status.

### POST /predict

Accepts a chest X-ray image and returns:

- Prediction
- Pneumonia probability

## Project Structure

Medical-Image-Classification/

├── app.py
├── requirements.txt

├── models/
│   └── pneumonia_mobilenetv2.keras

├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── final_results.csv

└── predictions/

## Important Note

This project is an academic machine-learning prototype and is not intended to provide clinical diagnosis or replace professional medical evaluation.

## Author

Mallikaarjun S
MCA - Artificial Intelligence and Machine Learning
