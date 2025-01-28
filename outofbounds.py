import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from abc import ABC, abstractmethod

class OutOfBoundsHandler(ABC):
    @abstractmethod
    def on_out_of_bounds(self, X, mask):
        """
        X (numpy.ndarray or pandas.DataFrame): The input data.
        mask (numpy.ndarray): Boolean mask for out-of-bounds values.

        Returns:
        numpy.ndarray: The processed data.
        """
        pass

class OutOfBoundsChecker(BaseEstimator, TransformerMixin):
    def __init__(self, lower_bound=None, upper_bound=None, on_out_of_bounds=None, default_behavior="drop"):
        """
        lower_bound (float or None): Lower bound. If None, no lower bound check.
        upper_bound (float or None): Upper bound. If None, no upper bound check.
        on_out_of_bounds (OutOfBoundsHandler or None): An instance of a class implementing the OutOfBoundsHandler interface.
        If provided, this function overrides the `default_behavior`.
        default_behavior (str): Default action when out-of-bounds data is detected, support 'drop'or 'raise_error'
        Ignored if `on_out_of_bounds` is specified.
        """

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.on_out_of_bounds = on_out_of_bounds
        self.default_behavior = default_behavior

        if self.default_behavior not in ["drop", "raise_error"]:
            raise ValueError("default_behavior must be either 'drop' or 'raise_error'")
        
        if on_out_of_bounds is not None and not isinstance(on_out_of_bounds, OutOfBoundsHandler):
            raise TypeError("on_out_of_bounds must implement the OutOfBoundsHandler interface.")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        mask = np.zeros_like(X, dtype=bool) 

        if self.lower_bound is not None:
            mask |= X < self.lower_bound

        if self.upper_bound is not None:
            mask |= X > self.upper_bound

        if self.lower_bound is None and self.upper_bound is None:
            warnings.warn(
                "No lower_bound or upper_bound provided. No out-of-bounds checks performed.",
                UserWarning
            )
            return X

        num_out_of_bounds = np.sum(mask)
        if num_out_of_bounds > 0:
            if self.on_out_of_bounds:
                return self.on_out_of_bounds.on_out_of_bounds(X, mask)
            else:
                if self.default_behavior == "drop":
                    warnings.warn(
                        f"{num_out_of_bounds} out-of-bounds values detected. Dropping them.",
                        UserWarning
                    )
                    return X[~mask]
                elif self.default_behavior == "raise_error":
                    raise ValueError(f"{num_out_of_bounds} out-of-bounds values detected in data.")

        return X