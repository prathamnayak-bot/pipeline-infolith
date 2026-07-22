import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from features import RatioFeatures, WorstMeanGapFeatures, SkewLogTransform

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

def test_ratio_features_shape():
    rf = RatioFeatures()
    out = rf.fit_transform(X)
    assert out.shape[0] == X.shape[0]
    assert out.shape[1] == len(rf.pairs)

def test_ratio_features_no_nan():
    rf = RatioFeatures()
    out = rf.fit_transform(X)
    assert not np.isnan(out).any()

def test_gap_features_correct_values():
    wg = WorstMeanGapFeatures(bases=["radius"])
    out = wg.fit_transform(X)
    expected = (X["worst radius"] - X["mean radius"]).values
    assert np.allclose(out.flatten(), expected)

def test_skew_log_no_negative_output_error():
    sk = SkewLogTransform(columns=["mean area"])
    out = sk.fit_transform(X)
    assert out.shape[0] == X.shape[0]

def test_transformers_stateless_no_leakage():
    # fitting on a subset shouldn't change output for the same rows
    rf = RatioFeatures()
    out_full = rf.fit_transform(X)
    out_subset = rf.fit_transform(X.iloc[:100])
    assert np.allclose(out_full[:100], out_subset)
