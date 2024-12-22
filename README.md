# Project Title

Child Mind Problematic Internet Use - Model Training

## Table of Contents

- [Introduction](#introduction)
- [Files Overview](#files-overview)
- [Detailed Code Descriptions](#detailed-code-descriptions)
- [Features](#features)
- [Results](#results)
- [Contact](#contact)

## Introduction

This project focuses on training machine learning models using data from the Kaggle Child Mind Problematic Internet Use dataset. The repository contains two key notebook files demonstrating different approaches to train and evaluate models for understanding problematic internet use patterns among children.

## Files Overview

### version12.ipynb

- **Description**: This notebook contains the initial version of the training pipeline. It includes basic preprocessing steps, feature engineering, and model training using standard techniques.
- **Key Features**:
  - Data exploration and cleaning.
  - Basic feature selection.
  - Model training and evaluation with initial hyperparameter settings.
  - Simple visualization of results.

### version18.ipynb

- **Description**: This notebook builds upon version 12 with optimized preprocessing, advanced feature engineering, and hyperparameter tuning. It introduces additional techniques to improve model performance.
- **Key Features**:
  - Enhanced preprocessing pipeline.
  - Advanced feature engineering methods.
  - Extensive hyperparameter tuning and cross-validation.
  - Detailed evaluation metrics and performance comparison.

## Detailed Code Descriptions

### version12.ipynb

1. **Data Exploration and Cleaning**:

   - Loads the dataset using pandas.
   - Performs basic exploratory data analysis (EDA) to understand the distribution of features and target variables.
   - Handles missing values and outliers using imputation techniques.

2. **Feature Engineering**:

   - Constructs new features based on domain knowledge.
   - Applies feature scaling to normalize the dataset for training.

3. **Model Training**:

   - Uses a basic machine learning algorithm (e.g., Logistic Regression or Decision Tree) to train the model.
   - Evaluates the model using accuracy, precision, recall, and F1 score metrics.

4. **Visualization**:

   - Includes plots to visualize feature importance and model performance.

### version18.ipynb

1. **Enhanced Preprocessing Pipeline**:

   - Implements advanced techniques for handling categorical variables, missing data, and outliers.
   - Utilizes libraries such as `sklearn` for robust preprocessing.

2. **Advanced Feature Engineering**:

   - Introduces polynomial features and interaction terms to capture complex relationships.
   - Applies dimensionality reduction techniques like PCA if needed.

3. **Hyperparameter Tuning**:

   - Uses Grid Search or Randomized Search for optimizing model parameters.
   - Incorporates cross-validation for robust performance evaluation.

4. **Model Training and Evaluation**:

   - Tests multiple algorithms (e.g., Random Forest, Gradient Boosting) for performance comparison.
   - Evaluates models with comprehensive metrics, including AUC-ROC and confusion matrices.

5. **Visualization and Insights**:

   - Provides advanced plots for model diagnostics and predictions.
   - Summarizes insights gained from the results, guiding future iterations.

## Features

- Comprehensive preprocessing and cleaning steps.
- Feature engineering tailored to the dataset.
- Model training using machine learning algorithms.
- Evaluation with appropriate metrics for classification problems.
- Insights from visualizations and performance analysis.

## Results

### Version 12 Results

- **Private: 0.465**
- **Public: 0.440**

### Version 18 Results

- **Private**: 0.403
- **Public: 0.485**

&#x20;Version 18 uses more advanced techniques, it focuses on improving public scores at the expense of a slight drop in private compared to Version 12.
