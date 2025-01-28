from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import Counter
import pandas as pd

class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 data_type="numerical",  # 明确传入数据类型：numerical 或 categorical
                 strategy=None, 
                 custom_func=None):
        """
        Parameters:
        data_type (str): Data type, "numerical" or "categorical".
        strategy (str): Missing value handling strategy. For numerical data, supports "mean", "most_common", or "custom";
                for categorical data, supports "most_common" or "custom".
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
        if self.data_type == "numerical" and self.strategy not in ["mean", "most_common", "custom"]:
            raise ValueError("Numerical strategy must be 'mean', 'most_common', or 'custom'")
        if self.data_type == "categorical" and self.strategy not in ["most_common", "custom"]:
            raise ValueError("Categorical strategy must be 'most_common' or 'custom'")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        isDataFrame = False
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            isDataFrame = True
        # Ensure X is a numpy array with object type to handle mixed data types
        X = np.asarray(X)

        if np.issubdtype(X.dtype, np.number):
            mask = np.isnan(X)
        else:
            mask = np.isin(X, [None, ""])

        # Handle missing values based on strategy
        if self.data_type == "numerical":
            result = self._handle_numerical(X, mask)
        elif self.data_type == "categorical":
            result = self._handle_categorical(X, mask)
            # Raise error if data type is not valid
        else:
            result = None
            raise ValueError("Invalid data_type provided.")

        # Convert back to DataFrame if input was DataFrame       
        if isDataFrame:
            print(pd.DataFrame(result, columns=columns))
            return pd.DataFrame(result, columns=columns)
        return result

    def _handle_numerical(self, X, mask):
        if self.strategy == "mean":
            fill_value = np.nanmean(X.astype(float))  # Ensure numerical calculation
        elif self.strategy == "most_common":
            fill_value = self._most_common(X[~mask])
        elif self.strategy == "custom" and callable(self.custom_func):
            fill_value = self.custom_func(X[~mask].astype(float))
        else:
            raise ValueError("Invalid numerical strategy or missing custom function.")

        X[mask] = fill_value
        return X

    def _handle_categorical(self, X, mask):
        if self.strategy == "most_common":
            fill_value = self._most_common(X[~mask])
        elif self.strategy == "custom" and callable(self.custom_func):
            fill_value = self.custom_func(X[~mask])
        else:
            raise ValueError("Invalid categorical strategy or missing custom function.")

        X[mask] = fill_value
        return X

    def _most_common(self, values):
        """
        Helper function to calculate the most common value in an array.
        """
        from collections import Counter
        counter = Counter(values)
        most_common_value, _ = counter.most_common(1)[0]
        return most_common_value
