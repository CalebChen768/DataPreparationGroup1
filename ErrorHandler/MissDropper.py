from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd



class MissDropper(BaseEstimator, TransformerMixin):
    def __init__(self, token_for_missing=None):
        self.token_for_missing = token_for_missing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.dropna()
        elif isinstance(X, np.ndarray):
            # Convert to DataFrame to drop missing values
            df = pd.DataFrame(X)
            return df.dropna().to_numpy()