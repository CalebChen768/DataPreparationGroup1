import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from ErrorHandler import *
from data import load_ratebeer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create test data
df = pd.DataFrame({
    "col_1": [1, 2, 4, np.nan, 5, 6, -7, 8, 9, 1000, 5], 
    "col_2": ["cat", "dog", None, "cat", "dog", "dog", "cat", "dog", "cat", "dog", "mice"],
    "col_3": ["Y","N","Y","N","Y","N","Y","N","Y","N","Y"],
    "col_4": [ "test I hey helo what calender insteresting",
              "This game is interesting. I strongly recommend it.",
              "This gmae is inteersting. I strongly recomend it.",
              "I am a student",
              "I am a student",
              "My name is John",
              "This is a test",
              "What can I do for you",
              "Haha, wonderful game",
              "asdmko fndosfhjdn safijdpwhauih fidnafpii uhwafi nedwb",
              "Do you like this game?"]
})

# Create a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("missing_values", MissingValueChecker(data_type="numerical", strategy="drop")),
            ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
            ("outliers", OutlierHandler(strategy="clip")),
            ("normalizer", ScaleAdjust(method="standard")),
            ("align_index", AlignTransformer(original_index=df.index)),
        ]), ["col_1"]),

        ("cat", Pipeline([
            ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
            ("bound_constrain", OutOfBoundsChecker(allowed_set=["cat", "dog"])),
            ("align_index", AlignTransformer(original_index=df.index)),
            ("onehot",OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in ["col_2"]], handle_unknown="ignore"))
        ]), ["col_2"]),

        ("cat2", Pipeline([
            ("onehot",OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in ["col_3"]], handle_unknown="ignore"))
        ]), ["col_3"]),
        
        ("text", Pipeline([
            ("gibberish", GibberishDetector(method="ngram")),
            ("align_index", AlignTransformer(original_index=df.index)),
        ]), ["col_4"]),
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X, columns=get_final_columns(preprocessor)), validate=False)),
    ("dropper", MissDropper()),
])


# def to_dataframe(X, ):
#     """
#     convert transformed data to DataFrame
#     """
#     # transformed_feature_names = transformer.get_feature_names_out(feature_names)
#     return pd.DataFrame(X, columns=)

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
        isBert = False
        for transformer in pipeline:
            if isinstance(transformer, OneHotEncoder):
                isOnehot = True
            if isinstance(transformer, BERTEmbeddingTransformer):
                isBert = True
        if isOnehot:
            # OneHotEncoder
            print(columns)
            if len(columns) != 0:
                for col in columns:
                    unique_categories = [list(set(df[i].unique())-{None, np.nan}) for i in [col]][0]
                    print("unique_categories", len(unique_categories))
                    generated_col_names = [f"{col}_{category}" for category in unique_categories]
                    final_columns.extend(generated_col_names)

        elif isBert:
            # for BERT transformer, keep original column names
            col = columns[0]
            extended_columns = [f"{col}_{i}" for i in range(768)]
            final_columns.extend(extended_columns)
        else:
            print("columns", columns)
            # for other transformers, keep original column names
            final_columns.extend(columns if isinstance(columns, list) else [columns])
        print("final_columns", final_columns)
    return final_columns




if __name__ == "__main__":
    df = load_ratebeer()
    df = df.head(10000)

    target_col = "beer/style"
    numerical_cols = ["review/appearance", "review/aroma", "review/palate"]
    # categorical_cols = ["beer/style"]
    categorical_cols = []
    text_cols = ["review/text"]

    df = df[[target_col] + numerical_cols + categorical_cols + text_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("missing_values", MissingValueChecker(data_type="numerical", strategy="mean")),
                ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
                ("outliers", OutlierHandler(strategy="clip")),
                ("normalizer", ScaleAdjust(method="standard")),
                ("align_index", AlignTransformer(original_index=df.index)),
            ]), numerical_cols),

            ("cat", Pipeline([
                ("missing_values", MissingValueChecker(data_type="categorical", strategy="most_common")),
                ("align_index", AlignTransformer(original_index=df.index)),
                ("onehot", OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in categorical_cols], handle_unknown="ignore"))
            ]), categorical_cols),

            ("text", Pipeline([
                ("gibberish", GibberishDetector(method="ngram")),
                ("bert_embedding", BERTEmbeddingTransformer()),
            ]), text_cols),
        ],
        remainder="drop"
    )


    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X, columns=get_final_columns(preprocessor)))),

        ("dropper", MissDropper()),
    ])

    model = DecisionTreeClassifier(random_state=42)

    import time

    # train a model
    np.random.seed(42)

    
  
    # print(df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split the dataset into training and testing sets
    pre_start = time.time()
    X = pipeline.fit_transform(X)
    pre_end = time.time()
    print(df.shape)
    print(f"Time taken to preprocess: {pre_end - pre_start:.2f} seconds")
    


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_fit = time.time()
    # Fit the model
    model.fit(X_train, y_train)
    end_fit = time.time()
    print(f"Time taken to fit: {end_fit - start_fit:.2f} seconds")

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print(f"Mean Squared Error: {mse:.2f}")
    # print(f"Mean Absolute Error: {mae:.2f}")
    # print(f"R^2 Score: {r2:.2f}")




    # print("============== Original Data ==============")
    # print(df)
    # print("\n============== Transformed Data ==============")
    # print(preprocessor.fit_transform(df))
    # print("\n============== Dropped NaN Data ==============")
    # print(pipeline.fit_transform(df))