import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeatures(BaseEstimator, TransformerMixin):
    """Adds ratio features between related mean/worst columns (e.g. concavity/area)."""
    def __init__(self, pairs=None):
        self.pairs = pairs or [
            ("mean concavity", "mean area"),
            ("worst concavity", "worst area"),
            ("mean concave points", "mean perimeter"),
            ("worst concave points", "worst perimeter"),
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X.copy()
        out = pd.DataFrame(index=X.index)
        for num, den in self.pairs:
            out[f"{num}_over_{den}"] = X[num] / (X[den] + 1e-9)
        return out.values

    def fit_transform(self, X, y=None):
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self.fit(X, y).transform(X)


class WorstMeanGapFeatures(BaseEstimator, TransformerMixin):
    """Adds (worst - mean) gap features — captures tumor heterogeneity."""
    def __init__(self, bases=None):
        self.bases = bases or [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave points", "symmetry", "fractal dimension"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for b in self.bases:
            out[f"{b}_gap"] = X[f"worst {b}"] - X[f"mean {b}"]
        return out.values


class SkewLogTransform(BaseEstimator, TransformerMixin):
    """Applies log1p to reduce skew on right-skewed columns; fit stores nothing (safe, stateless)."""
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = self.columns or list(X.columns)
        return self

    def transform(self, X):
        return np.log1p(X[self.columns_].clip(lower=0)).values
