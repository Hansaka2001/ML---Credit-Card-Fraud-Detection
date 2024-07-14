# Fraud Detection Using Random Forest Classifier

This repository contains a Jupyter notebook that demonstrates the use of a Random Forest classifier to detect fraudulent transactions. The notebook includes data preprocessing steps, model training, and evaluation metrics to assess the performance of the model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)

## Introduction

Fraud detection is a critical task in financial systems to prevent unauthorized transactions. This project uses a Random Forest classifier to identify fraudulent transactions from a dataset. The notebook provides a comprehensive walkthrough from data loading and preprocessing to model training and evaluation.

## Dataset

The dataset used in this project contains transaction data, with features that describe each transaction and a label indicating whether the transaction is fraudulent or not. The dataset should be in CSV format and include the following columns:
- Features describing the transaction
- A label column indicating fraud (1) or non-fraud (0)

## Dependencies

The following Python libraries are required to run the notebook:
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

You can install these dependencies using pip:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/fraud-detection.git
```

2. Navigate to the project directory:
```bash
cd fraud-detection
```

3. Open the Jupyter notebook:
```bash
jupyter notebook code.ipynb
```

4. Follow the steps in the notebook to load the dataset, preprocess the data, train the model, and evaluate its performance.

## Code Overview

The notebook is structured as follows:

1. **Data Loading and Preprocessing**:
    - Load the dataset.
    - Handle missing values, if any.
    - Split the data into training and testing sets.

2. **Model Training**:
    - Train a Random Forest classifier on the training data.

3. **Model Evaluation**:
    - Evaluate the model using accuracy, precision, recall, F1-score, and Matthews correlation coefficient.
    - Display a confusion matrix to visualize the model's performance.

## Results

The results section of the notebook includes:
- Accuracy, precision, recall, F1-score, and Matthews correlation coefficient of the trained model.
- A confusion matrix showing the true and predicted classes.

