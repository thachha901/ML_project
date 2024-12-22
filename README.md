# Project Title

Child Mind Problematic Internet Use - Model Training

## Table of Contents

- [Introduction](#introduction)
- [Files Overview](#files-overview)
- [Detailed Code Descriptions](#detailed-code-descriptions)
- [Results](#results)
- [Contact](#contact)

## Introduction

This project focuses on training machine learning models using data from the Kaggle Child Mind Problematic Internet Use dataset. The notebooks explore various preprocessing, feature engineering, and model training approaches to understand problematic internet use patterns among children.

## Files Overview

### version12.ipynb
- **Description**: This notebook implements a basic pipeline for data exploration, preprocessing, and model training.
- **Key Steps**:
  - Installing dependencies like `pytorch_tabnet`.
  - Loading the dataset from Kaggle.
  - Exploratory Data Analysis (EDA) with `pandas` and `matplotlib`.
  - Initial feature engineering and missing value handling.
  - Training LightGBM and XGBoost models using `StratifiedKFold` for cross-validation.

### version18.ipynb
- **Description**: This notebook builds upon version 12 by incorporating advanced techniques for preprocessing and modeling.
- **Key Steps**:
  - Implementing custom preprocessing functions.
  - Feature scaling with `MinMaxScaler` and dimensionality reduction using PCA.
  - Training a sparse autoencoder neural network with PyTorch for feature extraction.
  - Utilizing stacking and ensemble methods like VotingRegressor and StackingRegressor.
  - Hyperparameter optimization using Optuna for better performance.

## Detailed Code Descriptions

### version12.ipynb

1. **Data Loading**:
   - Datasets are loaded from Kaggle's directory structure, including train, test, and data dictionary files.
   - Example:
     ```python
     train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
     ```

2. **Exploratory Data Analysis (EDA)**:
   - Visualizes key features and target distributions.
   - Identifies missing values and applies simple imputations.

3. **Feature Engineering**:
   - Handles categorical variables using label encoding.
   - Standardizes numerical features with `StandardScaler`.

4. **Model Training**:
   - Implements LightGBM and XGBoost models.
   - Uses `StratifiedKFold` for validation to ensure robustness.
   - Metrics include Accuracy, Precision, Recall, and F1 Score.

### version18.ipynb

1. **Advanced Preprocessing**:
   - Builds a preprocessing pipeline with imputation and scaling.
   - Reduces feature dimensions using PCA.

2. **Autoencoder Neural Network**:
   - Trains a sparse autoencoder in PyTorch for feature extraction.
   - Code snippet:
     ```python
     class Autoencoder(nn.Module):
         def __init__(self, input_dim):
             super(Autoencoder, self).__init__()
             self.encoder = nn.Sequential(
                 nn.Linear(input_dim, 128),
                 nn.ReLU(),
                 nn.Linear(128, 64)
             )
             self.decoder = nn.Sequential(
                 nn.Linear(64, 128),
                 nn.ReLU(),
                 nn.Linear(128, input_dim)
             )
     ```

3. **Ensemble Modeling**:
   - Implements VotingRegressor with LightGBM, CatBoost, and XGBoost.
   - StackingRegressor combines GradientBoosting and RandomForest models.

4. **Hyperparameter Tuning**:
   - Uses Optuna to find the best parameters for LightGBM.
   - Example:
     ```python
     def objective(trial):
         param = {
             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
             'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1)
         }
         model = LGBMRegressor(**param)
         return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
     ```

## Results

### Version 12 Results
- **Private**: 0.465
- **Public**: 0.440

### Version 18 Results
- **Private**: 0.403
- **Public**: 0.485

Version 18 improves Public Score due to better feature extraction and optimization, though at a slight cost to Private Score.

