from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd



class MissDropper(BaseEstimator, TransformerMixin):
    def __init__(self, token_for_missing=None):
        self.token_for_missing = token_for_missing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        X = np.atleast_2d(X).T if X.ndim == 1 else X

        rows_with_none = np.any(X == self.token_for_missing, axis=1)

        return X[~rows_with_none]