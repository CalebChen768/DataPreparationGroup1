from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class AlignTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, original_index):
        self.original_index = original_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Align to the original index
        return pd.DataFrame(X, index=self.original_index).reindex(self.original_index)
