from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="clip", custom_func=None):
        """
        Parameters:
        strategy (str): Method to handle outliers, options:
            - "clip": Use IQR method to clip outliers to an acceptable range
            - "drop": Drop rows containing outliers
            - "custom": Use a custom handling function
        custom_func (callable): Custom function (must be provided when strategy="custom")
        """
        self.strategy = strategy
        self.custom_func = custom_func

        if self.strategy not in ["clip", "drop", "custom"]:
            raise ValueError("strategy must be 'clip', 'drop' or 'custom'")
        if self.strategy == "custom" and not callable(self.custom_func):
            raise ValueError("When strategy='custom', custom_func must be provided as a handling function")

    def fit(self, X, y=None):
        """
        Calculate IQR (Interquartile Range) during the fit stage, used for the clip strategy.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        # Calculate IQR range
        self.lower_bounds = {}
        self.upper_bounds = {}

        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - 1.5 * IQR
            self.upper_bounds[col] = Q3 + 1.5 * IQR

        return self

    def transform(self, X):
        """
        Handle outliers based on the chosen strategy
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        X_transformed = X.copy()

        for col in X_transformed.select_dtypes(include=[np.number]).columns:
            if self.strategy == "clip":
                # Clip outliers to a reasonable range
                X_transformed[col] = X_transformed[col].clip(
                    lower=self.lower_bounds[col], upper=self.upper_bounds[col]
                )
            elif self.strategy == "drop":
                # Find all outliers and drop rows containing these values
                mask = (X_transformed[col] < self.lower_bounds[col]) | (X_transformed[col] > self.upper_bounds[col])
                X_transformed = X_transformed[~mask]
            elif self.strategy == "custom":
                # Allow users to define their own outlier handling logic
                X_transformed[col] = self.custom_func(X_transformed[col])

        return X_transformed
