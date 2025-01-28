import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import pytest
from outofbounds import OutOfBoundsHandler, OutOfBoundsChecker


class MockHandler(OutOfBoundsHandler):
    def on_out_of_bounds(self, X, mask):
        X[mask] = 0
        return X

def test_out_of_bounds_checker():
    # No bounds provided
    checker = OutOfBoundsChecker()
    X = np.array([1, 2, 3])
    np.testing.assert_array_equal(checker.transform(X), X)

    # Lower bound only
    checker = OutOfBoundsChecker(lower_bound=2)
    X = np.array([1, 2, 3])
    np.testing.assert_array_equal(checker.transform(X), np.array([2, 3]))

    # Upper bound only
    checker = OutOfBoundsChecker(upper_bound=2)
    X = np.array([1, 2, 3])
    np.testing.assert_array_equal(checker.transform(X), np.array([1, 2]))

    # Both bounds
    checker = OutOfBoundsChecker(lower_bound=2, upper_bound=3)
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(checker.transform(X), np.array([2, 3]))

    # Custom handler
    handler = MockHandler()
    checker = OutOfBoundsChecker(lower_bound=2, upper_bound=3, on_out_of_bounds=handler)
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(checker.transform(X), np.array([0, 2, 3, 0]))

    # Raise error on out-of-bounds
    checker = OutOfBoundsChecker(lower_bound=2, upper_bound=3, default_behavior="raise_error")
    X = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        checker.transform(X)

    # Sklearn pipeline integration
    pipeline = Pipeline([
        ('out_of_bounds', OutOfBoundsChecker(lower_bound=2, upper_bound=3))
    ])
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(pipeline.transform(X), np.array([2, 3]))

    # No mask triggered
    checker = OutOfBoundsChecker(lower_bound=0, upper_bound=10)
    X = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(checker.transform(X), X)

    print("All tests passed!")

if __name__ == "__main__":
    test_out_of_bounds_checker()
