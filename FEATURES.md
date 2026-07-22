# Feature Engineering Documentation

## Project: Leak-Free Feature Engineering Pipeline
Dataset: Wisconsin Breast Cancer Dataset

---

## Overview

This project focuses on improving classification performance using feature engineering
while strictly avoiding data leakage. All transformations are implemented as
scikit-learn-compatible transformers and applied within a Pipeline to ensure that
statistics are learned only from training folds during cross-validation.

---

## Baseline

Model: Logistic Regression  
Preprocessing: StandardScaler  
ROC-AUC: ~0.97  

---

## Engineered Features

### 1. Ratio Features

**Features Created:**
- concavity_area_ratio = mean concavity / mean area
- perimeter_radius_ratio = mean perimeter / mean radius

**Motivation:**
Malignant tumors tend to exhibit irregular growth patterns. Ratios help normalize
scale differences and highlight structural abnormalities independent of size.

**Impact:**
Improves model sensitivity to shape irregularities.

---

### 2. Interaction Features

**Features Created:**
- radius_texture = mean radius × mean texture
- area_smoothness = mean area × mean smoothness

**Motivation:**
Certain relationships are not linearly separable in the original feature space.
Combining features allows the model to capture higher-order interactions.

**Impact:**
Improves separation between malignant and benign classes.

---

### 3. Log Transformation

**Applied to:**
All strictly positive features

**Motivation:**
Many features exhibit right-skewed distributions.
Log transformation reduces skewness and stabilizes variance.

**Impact:**
Improves linear model assumptions and robustness.

---

### 4. Outlier Clipping

**Method:**
IQR-based clipping (Q1 - 1.5×IQR, Q3 + 1.5×IQR)

**Motivation:**
Extreme values can distort model learning, especially in linear models.

**Leakage Prevention:**
Bounds are computed only during `fit()` on training data and reused in `transform()`.

**Impact:**
Reduces sensitivity to extreme observations.

---

## Leakage Prevention Strategy

All feature engineering steps are implemented inside a scikit-learn Pipeline.

This ensures:
- Each fold in cross-validation computes statistics independently
- No information from validation/test folds leaks into training

Incorrect approach (causes leakage):
```python
scaler.fit(X)  # WRONG
