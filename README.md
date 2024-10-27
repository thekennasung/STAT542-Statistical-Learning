# STAT542-Statistical-Learning

## Project1: Predict the Housing Prices in Ames

#### Real Estate Price Prediction Pipeline

This repository contains a machine learning pipeline for predicting real estate property prices using various regression models. The code handles data preprocessing, feature engineering, and model training. By utilizing techniques such as winsorization, standardization, and categorical encoding, the pipeline prepares data to ensure robust model performance.

#### Requirements

To set up the project, you will need the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`

You can install these packages using:

```bash
pip install pandas numpy scikit-learn
```
#### Code Overview

The main components of the code are:

- Feature Engineering: Prepares the dataset by removing unnecessary columns, applying transformations like winsorization to control outliers, and encoding categorical variables with one-hot encoding for compatibility with machine learning models.

- Modeling: Trains regression models on the processed data. Available models include:
  - GradientBoostingRegressor: A boosting-based regression model.
  - ElasticNet and ElasticNetCV: Models for linear regression with combined L1 and L2 regularization, which help prevent overfitting.

#### Key Function

- `feature_engineering(train_x, one_hot_encoders={})`: Transforms the dataset by:
  - Dropping irrelevant columns to focus on key predictive features.
  - Applying winsorization to cap extreme values in certain columns, which reduces the influence of outliers.
  - Encoding categorical features to numeric format for model compatibility.
    
- `winsorize(data, columns, threshold)`: Caps extreme values in specified columns according to the given threshold, which helps limit the impact of outliers.

#### Usage

1. Data Preprocessing: Load your dataset and preprocess it using the feature_engineering function.
   
2. Model Training: Train a model on the processed data. Examples of models in this pipeline include Gradient Boosting and ElasticNet regressors.
