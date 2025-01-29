from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class ScaleAdjust(BaseEstimator, TransformerMixin):
    def __init__(self, method="standard", feature_range=(0, 1)):
        """
        A custom transformer for scaling numerical data.

        Parameters:
        method (str): Scaling method, supports:
            - "standard" (Z-score standardization)
            - "minmax" (Min-Max normalization)
        feature_range (tuple): Only used for Min-Max scaling, defines the range of transformed values.
        """
        if method not in ["standard", "minmax"]:
            raise ValueError("method must be 'standard' or 'minmax'")

        self.method = method
        self.feature_range = feature_range
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None):
        """
        Compute the required statistics for scaling.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if self.method == "standard":
            self.mean_ = X.mean()
            self.std_ = X.std()
        elif self.method == "minmax":
            self.min_ = X.min()
            self.max_ = X.max()

        return self

    def transform(self, X):
        """
        Apply scaling transformation.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        X_transformed = X.copy()

        if self.method == "standard":
            X_transformed = (X - self.mean_) / self.std_
        elif self.method == "minmax":
            X_transformed = (X - self.min_) / (self.max_ - self.min_)
            X_transformed = X_transformed * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        return X_transformed

    def inverse_transform(self, X):
        """
        Reverse the transformation.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        X_inverse = X.copy()

        if self.method == "standard":
            X_inverse = (X * self.std_) + self.mean_
        elif self.method == "minmax":
            X_inverse = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            X_inverse = X_inverse * (self.max_ - self.min_) + self.min_

        return X_inverse
