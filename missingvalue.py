from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from collections import Counter


class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 data_type="numerical",  # numerical or categorical
                 strategy=None, 
                 custom_func=None):
        """
        Parameters:
        data_type (str): Data type, "numerical" or "categorical".
        strategy (str): Missing value handling strategy. For numerical data, supports "mean", "most_common", "drop" or "custom";
                for categorical data, supports "most_common", "drop" or "custom".
        custom_func (callable): Custom function to handle missing values (required when strategy="custom").
        """
        self.data_type = data_type
        if data_type == "numerical":
            self.missing_value = [np.nan, None]
        elif data_type == "categorical":
            self.missing_value = [np.nan, None, ""]
        self.strategy = strategy
        self.custom_func = custom_func

        # Check for valid input
        if self.data_type not in ["numerical", "categorical"]:
            raise ValueError("data_type must be 'numerical' or 'categorical'")
        if self.data_type == "numerical" and self.strategy not in ["mean", "most_common", "drop", "custom"]:
            raise ValueError("Numerical strategy must be 'mean', 'most_common', 'drop' or 'custom'")
        if self.data_type == "categorical" and self.strategy not in ["most_common", "drop", "custom"]:
            raise ValueError("Categorical strategy must be 'most_common', 'drop' or 'custom'")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if input is DataFrame
        if isinstance(X, pd.DataFrame):
            # Apply transformation column by column
            result = X.apply(self._transform_column, axis=0)
            # print(result)
            return result
        elif isinstance(X, pd.Series):
            # If a single Series is passed, treat it as a single column
            # print(result)
            return self._transform_column(X)
        else:
            raise TypeError("Input must be a pandas DataFrame or Series.")

    def _transform_column(self, col):
        """
        Apply missing value handling to a single column.
        """
        mask = col.isin(self.missing_value) | col.isnull()  # Detect missing values

        if col.dtype.kind in "bifc":  # If numerical data
            if self.data_type != "numerical":
                raise ValueError(f"Column {col.name} is numerical, but data_type='categorical' was provided.")
            return self._handle_numerical(col, mask)
        else:  # If categorical data
            if self.data_type != "categorical":
                raise ValueError(f"Column {col.name} is categorical, but data_type='numerical' was provided.")
            return self._handle_categorical(col, mask)

    def _handle_numerical(self, col, mask):
        """
        Handle missing values for a numerical column.
        """
        if self.strategy == "mean":
            fill_value = col[~mask].mean()  # Mean of non-missing values
        elif self.strategy == "most_common":
            fill_value = self._most_common(col[~mask])
        elif self.strategy == "drop":
            # change mask from NaN to None
            # print(col)
            # print(mask)
            # col = col.astype(object)
            # return col.mask(mask, None)
            return col.dropna()
        elif self.strategy == "custom" and callable(self.custom_func):
            fill_value = self.custom_func(col[~mask])
        else:
            raise ValueError("Invalid numerical strategy or missing custom function.")

        return col.fillna(fill_value)

    def _handle_categorical(self, col, mask):
        """
        Handle missing values for a categorical column.
        """
        if self.strategy == "most_common":
            fill_value = self._most_common(col[~mask])
        elif self.strategy == "drop":
            # change mask from NaN to None
            return col.mask(mask, None)
        elif self.strategy == "custom" and callable(self.custom_func):
            fill_value = self.custom_func(col[~mask])
        else:
            raise ValueError("Invalid categorical strategy or missing custom function.")

        return col.fillna(fill_value)

    def _most_common(self, values):
        """
        Helper function to calculate the most common value in a column.
        """
        counter = Counter(values)
        most_common_value, _ = counter.most_common(1)[0]
        return most_common_value
