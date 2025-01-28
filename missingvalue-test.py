import pandas as pd
import numpy as np
from missingvalue import MissingValueChecker

def test_missing_value_checker():
    # Create test data
    df = pd.DataFrame({
        "numerical_col": [1, 2, np.nan, 4],
        "categorical_col": ["cat", "dog", None, "cat"],
    })

    # Test numerical data: mean imputation
    checker = MissingValueChecker(data_type="numerical", strategy="mean")
    result = checker.transform(df["numerical_col"])
    print("Numerical with mean:")
    print(result)  # Should return [1.0, 2.0, 2.333..., 4.0]

    # Test numerical data: most common value imputation
    checker = MissingValueChecker(data_type="numerical", strategy="most_common")
    result = checker.transform(df["numerical_col"])
    print("Numerical with most common:")
    print(result)  # Should return [1.0, 2.0, 1.0, 4.0]

    # Test categorical data: most common value imputation
    checker = MissingValueChecker(data_type="categorical", strategy="most_common")
    result = checker.transform(df["categorical_col"])
    print("Categorical with most common:")
    print(result)  # Should return ["cat", "dog", "cat", "cat"]

    # Test custom function
    checker = MissingValueChecker(data_type="numerical", strategy="custom", custom_func=lambda x: x.min() - 1)
    result = checker.transform(df["numerical_col"])
    print("Numerical with custom function:")
    print(result)  # Should return [1.0, 2.0, 0.0, 4.0]

    # Test entire DataFrame
    checker = MissingValueChecker(data_type="numerical", strategy="mean")
    df_transformed = checker.transform(df)
    print("Transformed DataFrame:")
    print(df_transformed)

if __name__ == "__main__":
    test_missing_value_checker()
