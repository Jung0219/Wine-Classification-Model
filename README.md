# Wine Classification with Neural Networks

## Overview

This project implements a classification model for the wine dataset using neural networks. It showcases the use of dropout, batch normalization, and cross-entropy loss as part of my personal study on deep learning techniques with PyTorch.

## Features

### Data Preparation

- The dataset consists of chemical properties of wines and is split into training and testing sets using the `train_test_split` function from `sklearn`.

### Custom Dataset Loader

- A custom `WineDataset` class is implemented to handle loading and transforming the wine data into a format suitable for training.

### Model Architecture

- A feedforward neural network is created with:
  - Two hidden layers (32 and 16 neurons).
  - Dropout layers (50% probability) for regularization.
  - Batch normalization to stabilize learning.

### Training Process

- The model is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.002.
- Cross-entropy loss is used for calculating the model’s loss during training.

### Evaluation

- The trained model is evaluated on the test dataset, and the accuracy is calculated based on correct predictions.

## Output

- The test accuracy is printed at the end of the training process, providing insight into the model's performance.

## Technologies Used

- Python 3.10
- PyTorch
- NumPy
- Pandas
- scikit-learn
