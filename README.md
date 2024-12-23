# Project Title

Child Mind Problematic Internet Use - Model Training

## Table of Contents

- [Introduction](#introduction)
- [Files Overview](#files-overview)
- [Detailed Code Descriptions](#detailed-code-descriptions)
- [Features](#features)
- [Results](#results)

## Introduction

This project focuses on training machine learning models using data from the Kaggle Child Mind Problematic Internet Use dataset. The repository contains two key notebook files demonstrating different approaches to train and evaluate models for understanding problematic internet use patterns among children. While version 12 achieved a high private score, version 18 focuses on optimization and improved public performance.

## Files Overview

### version12.ipynb
- **Description**: This notebook contains the initial version of the training pipeline. It achieves a relatively high private score but lacks some advanced optimization techniques.
- **Key Features**:
  - Simple data preprocessing including handling missing values and basic transformations.
  - Training a LightGBM model with minimal hyperparameter tuning.
  - Performance focus on maximizing private leaderboard scores.

### version18.ipynb
- **Description**: This notebook builds upon version 12 with enhanced preprocessing, feature engineering, and extensive hyperparameter tuning. Although the private score decreases slightly, the public score improves significantly, demonstrating better generalization.
- **Key Features**:
  - Advanced data preprocessing such as outlier removal and categorical encoding.
  - Feature engineering to capture additional insights from the dataset.
  - Hyperparameter tuning using Bayesian optimization.
  - Model evaluation with detailed analysis of public and private scores.

## Detailed Code Descriptions

### version12.ipynb

1. **Data Preprocessing**:
   - Imputes missing values using mean imputation:
     ```python
     data.fillna(data.mean(), inplace=True)
     ```
   - Applies one-hot encoding for categorical features:
     ```python
     data = pd.get_dummies(data, columns=['categorical_column'])
     ```
   - Scales numeric features using Standard scaling:
     ```python
     scaler = StandardScaler()
     data_scaled = scaler.fit_transform(data)
     ```

2. **Model Training**:
   - Trains a LightGBM model with default parameters:
     ```python
     Light = LGBMRegressor(**Params, random_state=SEED, verbose=-1, n_estimators=300)
     XGB_Model = XGBRegressor(**XGB_Params)
     CatBoost_Model = CatBoostRegressor(**CatBoost_Params)
     TabNet_Model = TabNetWrapper(**TabNet_Params)

     voting_model = VotingRegressor(estimators=[
      ('lightgbm', Light),
      ('xgboost', XGB_Model),
      ('catboost', CatBoost_Model),
      ('tabnet', TabNet_Model)
     ],weights=[4.0,4.0,5.0,4.0]
     Submission,model = TrainML(voting_model,test))
        ```
        
3. **Performance**:
   - Focuses on achieving a high private score, resulting in:
     - **Private Score**: 0.465
     - **Public Score**: 0.440

### version18.ipynb

1. **Data Preprocessing**:
   - Handles missing values using KNN imputation:
     ```python
     imputer = KNNImputer(n_neighbors=5)
     data = imputer.fit_transform(data)
     ```
   - Encodes categorical variables using target encoding:
     ```python
     data['encoded_column'] = data['categorical_column'].map(target_encoding_dict)
     ```
   - Normalizes numeric features using Robust Scaling:
     ```python
     if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
      elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
      else:
        scaler = MinMaxScaler()
      data_scaled = scaler.fit_transform(data)
     ```

2. **Feature Engineering**:
   - Creates interaction features:
     ```python
     data['interaction'] = data['feature1'] * data['feature2']
     ```
   - Performs feature selection using feature importance:
     ```python
     important_features = model.feature_importances_ > threshold
     data_selected = data[:, important_features]
     ```

3. **Hyperparameter Tuning**:
   - Utilizes Bayesian optimization:
     ```python
     from bayes_opt import BayesianOptimization
     def lgb_eval(learning_rate, num_leaves):
         params = {'learning_rate': learning_rate, 'num_leaves': int(num_leaves)}
         model = LGBMClassifier(**params)
         return cross_val_score(model, X, y, cv=3).mean()
     optimizer = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 0.3), 'num_leaves': (20, 50)})
     optimizer.maximize()
     ```

4. **Performance**:
   - Aims to improve generalization, resulting in:
     - **Private Score**: 0.403
     - **Public Score**: 0.485

5. **Analysis**:
   - Explains the trade-off between private and public scores.
   - Highlights the improvements in public score due to better generalization techniques.

## Features

- Comprehensive preprocessing and cleaning steps.
- Feature engineering tailored to the dataset.
- Model training using LightGBM with advanced tuning techniques.
- Evaluation with both private and public leaderboard scores.

## Results

### Version 12 Results
- **Private Score**: 0.465
- **Public Score**: 0.440

### Version 18 Results
- **Private Score**: 0.403
- **Public Score**: 0.485

Although version 12 achieves a higher private score, version 18 demonstrates better public performance, which indicates improved generalization to unseen data.
