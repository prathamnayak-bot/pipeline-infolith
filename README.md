# pipeline-infolith
# Leak-Free Feature Engineering Pipeline

This project focuses on building a reliable machine learning pipeline with proper feature engineering while preventing data leakage.

The project uses the Breast Cancer Wisconsin dataset to perform binary classification. The main goal is to create reusable feature transformers, integrate them into a Scikit-Learn pipeline, and evaluate the model on unseen test data.

## Project Overview

The workflow includes:
- Data loading and preparation
- Custom feature engineering using Scikit-Learn transformers
- Building a leak-free ML pipeline
- Training a Logistic Regression classifier
- Evaluating performance using test-set metrics
- Unit testing feature transformers

## Files

- `notebook.ipynb` - Complete workflow including preprocessing, training, and evaluation
- `features.py` - Custom feature transformers used in the pipeline
- `test_features.py` - Tests to verify feature transformations
- `FEATURES.md` - Documentation of engineered features
- `requirements.txt` - Required Python packages

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Jupyter Notebook
- PyTest

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
