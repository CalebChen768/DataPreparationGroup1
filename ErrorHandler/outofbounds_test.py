import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import pytest
from outofbounds import OutOfBoundsHandler, OutOfBoundsChecker
import pandas as pd
from MissDropper import MissDropper

class MockHandler(OutOfBoundsHandler):
    def on_out_of_bounds(self, X, mask):
        X[mask] = 0
        return X

def test_out_of_bounds_checker():
    # No bounds provided
    checker = Pipeline([("1", OutOfBoundsChecker()), ("2", MissDropper())])
    X = np.array([1, 2, 3])
    np.testing.assert_array_equal(checker.fit_transform(X), X.reshape(-1, 1))

    # Lower bound only
    checker = Pipeline([("1", OutOfBoundsChecker(lower_bound=2)), ("2", MissDropper())])
    X = np.array([1, 2, 3])
    np.testing.assert_array_equal(checker.fit_transform(X), np.array([2, 3]).reshape(-1, 1))

    # Upper bound only
    checker = Pipeline([("1", OutOfBoundsChecker(upper_bound=2)), ("2", MissDropper())])
    X = np.array([1, 2, 3])
    np.testing.assert_array_equal(checker.fit_transform(X), np.array([1, 2]).reshape(-1, 1))

    # Both bounds
    checker = Pipeline([("1", OutOfBoundsChecker(lower_bound=2, upper_bound=3)), ("2", MissDropper())])
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(checker.fit_transform(X), np.array([2, 3]).reshape(-1, 1))

    # Custom handler
    handler = MockHandler()
    checker = OutOfBoundsChecker(lower_bound=2, upper_bound=3, on_out_of_bounds=handler)
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(checker.fit_transform(X), np.array([0, 2, 3, 0]).reshape(-1, 1))

    # Raise error on out-of-bounds
    checker = OutOfBoundsChecker(lower_bound=2, upper_bound=3, default_behavior="raise_error")
    X = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        checker.fit_transform(X)

    # No mask triggered
    checker = OutOfBoundsChecker(lower_bound=0, upper_bound=10)
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(checker.fit_transform(X), X.reshape(-1, 1))

    # Pandas with categorical data, within pipeline
    allow_set = {"Kitchen", "Beauty"}
    df = pd.DataFrame([
                        {"Category":"Kitchen", "Price": 1, "vote":2},
                        {"Category":"Kitchen", "Price": -1, "vote":1},
                        {"Category":"Ktchen", "Price": 1, "vote":1},
                        {"Category":"Beauty", "Price": 1, "vote":1},
                       ])
    pipeline = Pipeline([(
        "pro",
        
        ColumnTransformer(
        [("num", OutOfBoundsChecker(lower_bound=0, upper_bound=2), ["Price", "vote"]),
        ("text", OutOfBoundsChecker(allowed_set=allow_set), ["Category"])]
    )),
    (
        "drop",
        MissDropper()
    )
    ])
    
    df = pipeline.fit_transform(df)
    np.array_equal(df, np.array([(1, 2, "Kitchen"),
                                  (1, 1, "Beauty")]))

    print("All tests passed!")

if __name__ == "__main__":
    test_out_of_bounds_checker()
