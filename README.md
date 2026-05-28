# CIFAR-10 Image Classification with Convolutional Neural Networks

A supervised multi-class image classification project comparing two CNN architectures on the CIFAR-10 dataset. Built in Python using TensorFlow/Keras as the final project for *Introduction to Machine Learning* (MTH 3320) at St. John's University.

## Overview

The goal of this project was to build and compare convolutional neural networks for classifying 32×32 color images into 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Two models were trained and evaluated to study how architectural choices — specifically network depth and dropout regularization — affect both training behavior and generalization to unseen data.

## Dataset

- **CIFAR-10**: 60,000 32×32 RGB images across 10 classes
- 50,000 training images / 10,000 test images
- Pixel values normalized to [0, 1]
- Labels one-hot encoded for multi-class classification

## Models

**Model 1 — Baseline CNN**
- 2 convolutional layers (32, 64 filters) with ReLU activation
- 2 max-pooling layers
- 1 dense hidden layer (64 units)
- Softmax output layer
- ~167K trainable parameters

**Model 2 — Deeper CNN with Dropout**
- 3 convolutional layers (32, 64, 64 filters) with ReLU activation
- 2 max-pooling layers
- 1 dense hidden layer (64 units) + Dropout (0.5)
- Softmax output layer
- ~122K trainable parameters

Both models were trained for 10 epochs using the Adam optimizer with categorical cross-entropy loss and a 20% validation split.

## Results

| Model | Test Loss | Test Accuracy |
|-------|-----------|---------------|
| CNN Model 1 | 1.0081 | 66.6% |
| CNN Model 2 | 0.8875 | 68.7% |

**Key findings:**
- Model 1 showed clear signs of overfitting — training accuracy continued to climb while validation accuracy plateaued and validation loss began to rise in later epochs.
- Adding a third convolutional layer and dropout in Model 2 closed the train/validation gap and produced more stable learning curves.
- Both models performed well on visually distinct classes (ship, automobile, truck, frog) and struggled with visually similar animal classes (cat, dog, bird, deer) — consistent with the inherent difficulty of low-resolution animal classification.

## What I'd Improve Next

Honest assessment of the limits of this project and what a follow-up would look like:

1. **Data augmentation** — random flips, crops, and rotations would meaningfully boost accuracy and reduce overfitting more than dropout alone.
2. **Batch normalization** — stabilizes training and typically allows for better convergence on this kind of architecture.
3. **More epochs with early stopping** — 10 epochs is short; with augmentation, training longer would help.
4. **Deeper architectures or transfer learning** — pretrained models (ResNet, VGG) could push test accuracy above 90% on CIFAR-10.
5. **Per-class performance tuning** — the model's weakness on animal classes is a known CIFAR-10 challenge worth targeting directly with class-weighted loss or targeted augmentation.

## Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy, pandas
- scikit-learn (classification report, confusion matrix)
- matplotlib

## Files

- `Ashton_ML_Final.ipynb` — full notebook with code, output, and written analysis
- `Ashton_ML_Final.pdf` — rendered PDF version for easy viewing
