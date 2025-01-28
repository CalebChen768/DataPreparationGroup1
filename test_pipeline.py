import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from missingvalue import MissingValueChecker
from outlier import OutlierHandler
from alignTransformer import AlignTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

# Create test data
df = pd.DataFrame({
    "col 1": [1, 2, np.nan, 4, 5, 6, -7, 8, 9, 1000], 
    "col 2": ["cat", "dog", None, "cat", "dog", "dog", "cat", "dog", "cat", "dog"],
})

# Create a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("missing_values", MissingValueChecker(data_type="numerical", strategy="mean")),
            ("outliers", OutlierHandler(strategy="clip")),
            ("normalizer", StandardScaler()),
            ("align_index", AlignTransformer(original_index=df.index)),
        ]), ["col 1"]),

        ("cat", Pipeline([
            ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
            ("align_index", AlignTransformer(original_index=df.index)),
        ]), ["col 2"]),
    ],
    remainder="passthrough"
)


if __name__ == "__main__":
    print("============== Original Data ==============")
    print(df)
    print("\n============== Transformed Data ==============")
    print(preprocessor.fit_transform(df))