# Heart Disease Prediction & Model Explainability

A machine learning project for heart disease prediction using multiple classification models, hyperparameter tuning, model evaluation, threshold optimization, and SHAP-based explainability.

## Overview

This project investigates the use of machine learning for binary heart disease prediction using clinical and demographic features.

Three classification approaches are developed and compared:

* **K-Nearest Neighbors (KNN)**
* **Random Forest**
* **XGBoost**

The workflow includes preprocessing, stratified cross-validation, hyperparameter tuning, model selection with overfitting considerations, test-set evaluation, F1-optimal threshold selection, and model explainability using SHAP.

> **Note:** This project is intended for machine learning research and portfolio demonstration. It is not a clinical diagnostic system.

## Dataset

The project uses a heart disease dataset containing:

* **303 patient records**
* **13 input features**
* **1 binary target variable**

The features include demographic, clinical, and exercise-related variables such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Slope
* Number of major vessels
* Thalassemia

The target variable represents the presence or absence of heart disease.

## Methodology

### 1. Data Exploration

The notebook performs:

* Dataset inspection
* Descriptive statistics
* Data type and missing-value inspection
* Correlation analysis
* Target distribution analysis

### 2. Preprocessing

The preprocessing pipeline separates numerical and categorical variables.

**Numerical features:**

* StandardScaler

**Categorical features:**

* OneHotEncoder

The preprocessing is implemented using a Scikit-learn `ColumnTransformer` and integrated into model pipelines.

### 3. Train-Test Split

The dataset is divided into:

* **80% training**
* **20% test**

Stratification is used to preserve the target-class distribution.

Five-fold **Stratified Cross-Validation** is used during model selection.

### 4. Model Development

#### K-Nearest Neighbors

Hyperparameters explored include:

* Number of neighbors
* Distance metric
* Weighting strategy

The selected configuration was:

* `n_neighbors = 9`
* `p = 1`
* `weights = uniform`

Test performance at the default threshold:

| Metric    |     Score |
| --------- | --------: |
| ROC-AUC   | **0.887** |
| F1        | **0.800** |
| Precision | **0.714** |
| Recall    | **0.909** |

#### Random Forest

Hyperparameters explored include:

* Number of estimators
* Maximum tree depth
* Minimum samples for splitting
* Class weighting

Selected configuration:

* `n_estimators = 400`
* `max_depth = 5`
* `min_samples_split = 10`
* `class_weight = balanced`

At the default threshold of 0.5:

| Metric    |     Score |
| --------- | --------: |
| ROC-AUC   | **0.915** |
| F1        | **0.817** |
| Precision | **0.763** |
| Recall    | **0.879** |

After selecting the F1-optimal probability threshold:

| Metric            |     Score |
| ----------------- | --------: |
| ROC-AUC           | **0.915** |
| Average Precision | **0.934** |
| F1                | **0.866** |
| Precision         | **0.853** |
| Recall            | **0.879** |

#### XGBoost

Hyperparameters explored include:

* Number of estimators
* Learning rate
* Maximum tree depth
* Subsampling
* Column sampling
* Positive-class weighting

Selected configuration:

* `n_estimators = 200`
* `max_depth = 3`
* `learning_rate = 0.01`
* `subsample = 0.8`
* `colsample_bytree = 0.8`
* `scale_pos_weight ≈ 0.833`

At the default threshold of 0.5:

| Metric    |     Score |
| --------- | --------: |
| ROC-AUC   | **0.904** |
| F1        | **0.841** |
| Precision | **0.806** |
| Recall    | **0.879** |

After F1-optimal threshold selection:

| Metric            |     Score |
| ----------------- | --------: |
| ROC-AUC           | **0.904** |
| Average Precision | **0.917** |
| F1                | **0.853** |
| Precision         | **0.762** |
| Recall            | **0.970** |

## Model Comparison

Based on the test-set results, **Random Forest achieved the highest ROC-AUC (0.915)** among the evaluated models.

| Model         | Test ROC-AUC |  F1 @ 0.5 | Precision @ 0.5 | Recall @ 0.5 |
| ------------- | -----------: | --------: | --------------: | -----------: |
| KNN           |        0.887 |     0.800 |           0.714 |        0.909 |
| Random Forest |    **0.915** |     0.817 |           0.763 |        0.879 |
| XGBoost       |        0.904 | **0.841** |       **0.806** |        0.879 |

Threshold optimization demonstrates that model performance can change substantially depending on the classification threshold and the desired balance between precision and recall.

## Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to investigate how individual features contribute to model predictions.

For both Random Forest and XGBoost, the project generates:

* **SHAP beeswarm plots** for feature-level impact and distribution
* **SHAP bar plots** for global feature importance
* **SHAP waterfall plots** for individual prediction explanations

This provides an interpretable view of how different clinical features influence the model's predictions.

## Evaluation & Visualization

The project includes several evaluation and diagnostic techniques:

* ROC curves
* Precision-Recall curves
* Confusion matrices
* Learning curves
* Cross-validation
* Train-validation performance gap analysis
* F1-optimal threshold selection
* SHAP explainability

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Jupyter Notebook

## Repository Structure

```text
heart-disease-ml-explainability/
│
├── heart_disease_explainability.ipynb
└── README.md
```

## Key Takeaways

* Multiple machine learning algorithms were evaluated for heart disease prediction.
* Random Forest achieved the highest test ROC-AUC among the evaluated models.
* Hyperparameter tuning was combined with cross-validation and train-validation gap analysis.
* Probability threshold optimization improved the F1 score for both Random Forest and XGBoost.
* SHAP was used to improve the interpretability of model predictions.

## Future Improvements

Potential extensions include:

* External validation on an independent dataset
* Nested cross-validation
* Calibration analysis
* More systematic feature selection
* Comparison with additional models such as Logistic Regression and SVM
* Evaluation on larger and more diverse clinical datasets
* More rigorous uncertainty and confidence analysis

## Disclaimer

This repository is a machine learning portfolio project and should not be used for medical diagnosis or clinical decision-making.

