import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from missingvalue import MissingValueChecker
from outlier import OutlierHandler
from alignTransformer import AlignTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from MissDropper import MissDropper
from outofbounds import OutOfBoundsChecker
from functools import partial


# Create test data
df = pd.DataFrame({
    "col_1": [1, 2, 4, np.nan, 5, 6, -7, 8, 9, 1000, 5], 
    "col_2": ["cat", "dog", None, "cat", "dog", "dog", "cat", "dog", "cat", "dog", "mice"],
})

# Create a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("missing_values", MissingValueChecker(data_type="numerical", strategy="mean")),
            ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
            ("outliers", OutlierHandler(strategy="clip")),
            ("normalizer", StandardScaler()),
            ("align_index", AlignTransformer(original_index=df.index)),
        ]), ["col_1"]),

        ("cat", Pipeline([
            ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
            ("bound_constrain", OutOfBoundsChecker(allowed_set=["cat", "dog"])),
            ("align_index", AlignTransformer(original_index=df.index)),
            ("onehot",OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in ["col_2"]], handle_unknown="ignore"))
        ]), ["col_2"]),
    ],
    remainder="passthrough"
)

def to_dataframe(X, transformer, feature_names):
    """
    convert transformed data to DataFrame
    """
    transformed_feature_names = transformer.get_feature_names_out(feature_names)
    return pd.DataFrame(X, columns=transformed_feature_names)

def get_final_columns(column_transformer):
    """
    Iterate through ColumnTransformer to get the final transformed column names.
    - If it is OneHotEncoder, calculate categories and get new column names
    - For other transformers, keep the original column names
    """
    final_columns = []
    # column_transformer.fit(df)
    for name, pipeline, columns in column_transformer.transformers_:
        isOnehot = False

        for transformer in pipeline:
            if isinstance(transformer, OneHotEncoder):
                isOnehot = True
        if isOnehot:
            # OneHotEncoder
            for col in columns:
                unique_categories = [list(set(df[i].unique())-{None, np.nan}) for i in ["col_2"]][0]
                print("unique_categories", unique_categories)
                generated_col_names = [f"{col}_{category}" for category in unique_categories]
                final_columns.extend(generated_col_names)
        else:
            # for other transformers, keep original column names
            final_columns.extend(columns if isinstance(columns, list) else [columns])
    
    return final_columns



pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X, columns=get_final_columns(preprocessor)), validate=False)),
    ("dropper", MissDropper()),

])

if __name__ == "__main__":
    print("============== Original Data ==============")
    print(df)
    print("\n============== Transformed Data ==============")
    print(preprocessor.fit_transform(df))
    print("\n============== Dropped NaN Data ==============")
    print(pipeline.fit_transform(df))