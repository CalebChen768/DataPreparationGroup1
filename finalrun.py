import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from ErrorHandler import *
from data import load_ratebeer, DataErrorAdder
# import tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
# import LinearRegression
from sklearn.linear_model import LinearRegression
# import RandomForestRegressor
# import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.dummy import DummyClassifier, DummyRegressor

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
            if isinstance(transformer, BERTEmbeddingTransformer2):
                isBert = True
        if isOnehot:
            # OneHotEncoder
            print(columns)
            if len(columns) != 0:
                for col in columns:
                    unique_categories = [list(set(A[i].unique())-{None, np.nan}) for i in [col]][0]
                    print("unique_categories", len(unique_categories))
                    generated_col_names = [f"{col}_{category}" for category in unique_categories]
                    final_columns.extend(generated_col_names)

        elif isBert:
            # for BERT transformer, keep original column names
            if len(columns) != 0:
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
    # without manually add errors
    # A = load_ratebeer("sampled_dataframe2.csv")
    # # with manually add errors
    # B = pd.read_csv("sampled_dataframe3.csv")
    # C = pd.read_csv("sampled_dataframe3.csv")

    # B['review/appearance'] = B['review/appearance'].astype(float)
    # B['review/aroma'] = B['review/aroma'].astype(float)
    # B['review/palate'] = B['review/palate'].astype(float)
    # B['review/taste'] = B['review/taste'].astype(float)
    # B['review/overall'] = B['review/overall'].astype(float)
    # B['beer/ABV'] = B['beer/ABV'].astype(float)

    # C['review/appearance'] = C['review/appearance'].astype(float)
    # C['review/aroma'] = C['review/aroma'].astype(float)
    # C['review/palate'] = C['review/palate'].astype(float)
    # C['review/taste'] = C['review/taste'].astype(float)
    # C['review/overall'] = C['review/overall'].astype(float)
    # C['beer/ABV'] = C['beer/ABV'].astype(float)

    A = load_ratebeer("sampled_dataframe2.csv")
    B = A.copy()
    # add error
    error_adder = DataErrorAdder("config.yaml")
    B = error_adder.add_errors(B)
    C = B.copy()


    num_data = 5000
    A = A.head(num_data)
    B = B.head(num_data)
    C = C.head(num_data)
    
    numerical_cols = ["review/appearance", "review/aroma", "review/palate", "review/taste"]
    target_col = "review/overall"
    categorical_cols = []
    text_cols = ["review/text"]

    # text_cols = []
    # keep only the columns that are used
    A = A[[target_col] + numerical_cols + categorical_cols + text_cols]
    B = B[[target_col] + numerical_cols + categorical_cols + text_cols]
    C = C[[target_col] + numerical_cols + categorical_cols + text_cols]


    # pipeline for A
    preprocessor_A = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("missing_values", MissingValueChecker(data_type="numerical", strategy="drop")),
                # ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
                # ("outliers", OutlierHandler(strategy="clip")),
                ("normalizer", ScaleAdjust(method="standard")),
                ("align_index", AlignTransformer(original_index=A.index)),
            ]), numerical_cols),

            ("cat", Pipeline([
                ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
                ("align_index", AlignTransformer(original_index=A.index)),
                ("onehot", OneHotEncoder(categories=[list(set(A[i].unique())-{None, np.nan}) for i in categorical_cols], handle_unknown="ignore"))
            ]), categorical_cols),

            ("text", Pipeline([
                # ("gibberish", GibberishDetector(method="ngram")),
                ("bert_embedding", BERTEmbeddingTransformer2()),
                # ("tf_idf", TfidfVectorizer()),
                ("align_index", AlignTransformer(original_index=A.index)),
                ("missing_text", MissingValueChecker(data_type="numerical", strategy="drop")),
                ("align_index2", AlignTransformer(original_index=A.index)),
            ]), text_cols),
        ],
        remainder="drop"
    )


    # pipeline for B
    preprocessor_B = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("missing_values", MissingValueChecker(data_type="numerical", strategy="drop")),
                # ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
                # ("outliers", OutlierHandler(strategy="clip")),
                ("normalizer", ScaleAdjust(method="standard")),
                ("align_index", AlignTransformer(original_index=B.index)),
            ]), numerical_cols),

            ("cat", Pipeline([
                ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
                ("align_index", AlignTransformer(original_index=B.index)),
                ("onehot", OneHotEncoder(categories=[list(set(B[i].unique())-{None, np.nan}) for i in categorical_cols], handle_unknown="ignore"))
            ]), categorical_cols),

            ("text", Pipeline([
                # ("gibberish", GibberishDetector(method="ngram")),
                ("bert_embedding", BERTEmbeddingTransformer2()),
                # ("tf_idf", TfidfVectorizer()),
                ("align_index", AlignTransformer(original_index=B.index)),
                ("missing_text", MissingValueChecker(data_type="numerical", strategy="drop")),
                ("align_index2", AlignTransformer(original_index=B.index)),
            ]), text_cols),
        ],
        remainder="drop"
    )

    # pipeline for C
    preprocessor_C = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("missing_values", MissingValueChecker(data_type="numerical", strategy="mean")),
                ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
                ("outliers", OutlierHandler(strategy="clip")),
                ("normalizer", ScaleAdjust(method="standard")),
                ("align_index", AlignTransformer(original_index=C.index)),
            ]), numerical_cols),

            ("cat", Pipeline([
                ("missing_values", MissingValueChecker(data_type="categorical", strategy="most_common")),
                ("align_index", AlignTransformer(original_index=C.index)),
                ("onehot", OneHotEncoder(categories=[list(set(C[i].unique())-{None, np.nan}) for i in categorical_cols], handle_unknown="ignore"))
            ]), categorical_cols),

            ("text", Pipeline([
                # ("gibberish", GibberishDetector(method="ngram")),
                ("gibberish", TwoStepGibberishDetector()),
                ("bert_embedding", BERTEmbeddingTransformer2()),
                # ("tf_idf", TfidfVectorizer()),
                ("align_index", AlignTransformer(original_index=C.index)),
                ("missing_text", MissingValueChecker(data_type="numerical", strategy="most_common")),
                ("align_index2", AlignTransformer(original_index=C.index)),
            ]), text_cols),
        ],
        remainder="drop"
    )


    # pipeline for A
    pipeline_A = Pipeline([
        ("preprocessor", preprocessor_A),
        ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X))),
        ("dropper", MissDropper()),
    ])
   
    # pipeline for B
    pipeline_B = Pipeline([
        ("preprocessor", preprocessor_B),
        ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X))),
        ("dropper", MissDropper()),
    ])

    # pipeline for C
    pipeline_C = Pipeline([
        ("preprocessor", preprocessor_C),
        ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X))),
        ("dropper", MissDropper()),
    ])

    # train a model
    np.random.seed(42)

    X_A = A.drop(columns=[target_col])
    y_A = A[target_col]
    X_B = B.drop(columns=[target_col])  
    y_B = B[target_col]
    X_C = C.drop(columns=[target_col])
    y_C = C[target_col]

    # X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
    # X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
    # X_C_train, X_C_test, y_C_train, y_C_test = train_test_split(X_C, y_C, test_size=0.2, random_state=42)

    print("==========A transform==========")
    X_A = pipeline_A.fit_transform(X_A)
    y_A = y_A.loc[X_A.index]
    print("==========B transform==========")
    X_B = pipeline_B.fit_transform(X_B)
    y_B = y_B.loc[X_B.index]
    print("==========C transform==========")
    X_C = pipeline_C.fit_transform(X_C)
    y_C = y_C.loc[X_C.index]

    # print("==========A transform==========")
    # X_A_train = pipeline_A.fit_transform(X_A_train)
    # y_A_train = y_A_train.loc[X_A_train.index]
    # X_A_test = pipeline_A.transform(X_A_test)
    # y_A_test = y_A_test.loc[X_A_test.index]
    # print("==========B transform==========")
    # X_B_train = pipeline_B.fit_transform(X_B_train)
    # y_B_train = y_B_train.loc[X_B_train.index]
    # X_B_test = pipeline_B.transform(X_B_test)
    # y_B_test = y_B_test.loc[X_B_test.index]
    # print("==========C transform==========")
    # X_C_train = pipeline_C.fit_transform(X_C_train)
    # y_C_train = y_C_train.loc[X_C_train.index]
    # X_C_test = pipeline_C.transform(X_C_test)
    # y_C_test = y_C_test.loc[X_C_test.index]

    # # reindex all
    # X_A_train = X_A_train.reindex(X_A_train.index)
    # y_A_train = y_A_train.reindex(y_A_train.index)
    # X_A_test = X_A_test.reindex(X_A_test.index)
    # y_A_test = y_A_test.reindex(y_A_test.index)

    # X_B_train = X_B_train.reindex(X_B_train.index)
    # y_B_train = y_B_train.reindex(y_B_train.index)
    # X_B_test = X_B_test.reindex(X_B_test.index)
    # y_B_test = y_B_test.reindex(y_B_test.index)

    # X_C_train = X_C_train.reindex(X_C_train.index)
    # y_C_train = y_C_train.reindex(y_C_train.index)
    # X_C_test = X_C_test.reindex(X_C_test.index)
    



    # XYA  = pd.concat([X_A, y_A], axis=1)
    # XYB  = pd.concat([X_B, y_B], axis=1)
    # XYC  = pd.concat([X_C, y_C], axis=1)
    # XYA.to_csv("XYA.csv", index=False)
    # XYB.to_csv("XYB.csv", index=False)
    # XYC.to_csv("XYC.csv", index=False)

    print("==========split dataset==========")
    
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
    X_C_train, X_C_test, y_C_train, y_C_test = train_test_split(X_C, y_C, test_size=0.2, random_state=42)

    # model_A = DecisionTreeRegressor(random_state=42)
    # model_B = DecisionTreeRegressor(random_state=42)
    # model_C = DecisionTreeRegressor(random_state=42)
    
    model_A = RandomForestRegressor(random_state=42)
    model_B = RandomForestRegressor(random_state=42)
    model_C =  RandomForestRegressor(random_state=42)
    print("==========fit model==========")
    model_A.fit(X_A_train, y_A_train)
    model_B.fit(X_B_train, y_B_train)
    model_C.fit(X_C_train, y_C_train)

    print("==========predict model==========")
    # model on dataset
    y_A_on_A = model_A.predict(X_A_test)
    y_A_on_B = model_A.predict(X_B_test)
    y_A_on_C = model_A.predict(X_C_test)
    y_B_on_A = model_B.predict(X_A_test)
    y_B_on_B = model_B.predict(X_B_test)
    y_B_on_C = model_B.predict(X_C_test)
    y_C_on_A = model_C.predict(X_A_test)
    y_C_on_B = model_C.predict(X_B_test)
    y_C_on_C = model_C.predict(X_C_test)


    print("==========calculate errors==========")
    mse_A_on_A = mean_squared_error(y_A_test, y_A_on_A)
    mae_A_on_A = mean_absolute_error(y_A_test, y_A_on_A)
    r2_A_on_A = r2_score(y_A_test, y_A_on_A)

    mse_A_on_B = mean_squared_error(y_B_test, y_A_on_B)
    mae_A_on_B = mean_absolute_error(y_B_test, y_A_on_B)
    r2_A_on_B = r2_score(y_B_test, y_A_on_B)

    mse_A_on_C = mean_squared_error(y_C_test, y_A_on_C)
    mae_A_on_C = mean_absolute_error(y_C_test, y_A_on_C)
    r2_A_on_C = r2_score(y_C_test, y_A_on_C)

    mse_B_on_A = mean_squared_error(y_A_test, y_B_on_A)
    mae_B_on_A = mean_absolute_error(y_A_test, y_B_on_A)
    r2_B_on_A = r2_score(y_A_test, y_B_on_A)

    mse_B_on_B = mean_squared_error(y_B_test, y_B_on_B)
    mae_B_on_B = mean_absolute_error(y_B_test, y_B_on_B)
    r2_B_on_B = r2_score(y_B_test, y_B_on_B)

    mse_B_on_C = mean_squared_error(y_C_test, y_B_on_C)
    mae_B_on_C = mean_absolute_error(y_C_test, y_B_on_C)
    r2_B_on_C = r2_score(y_C_test, y_B_on_C)

    mse_C_on_A = mean_squared_error(y_A_test, y_C_on_A)
    mae_C_on_A = mean_absolute_error(y_A_test, y_C_on_A)
    r2_C_on_A = r2_score(y_A_test, y_C_on_A)

    mse_C_on_B = mean_squared_error(y_B_test, y_C_on_B)
    mae_C_on_B = mean_absolute_error(y_B_test, y_C_on_B)
    r2_C_on_B = r2_score(y_B_test, y_C_on_B)

    mse_C_on_C = mean_squared_error(y_C_test, y_C_on_C)
    mae_C_on_C = mean_absolute_error(y_C_test, y_C_on_C)
    r2_C_on_C = r2_score(y_C_test, y_C_on_C)

    

    print("==========result of A on A==========")
    print(f"Mean Squared Error: {mse_A_on_A:.2f}")
    print(f"Mean Absolute Error: {mae_A_on_A:.2f}")
    print(f"R^2 Score: {r2_A_on_A:.2f}")

    print("==========result of A on B==========")
    print(f"Mean Squared Error: {mse_A_on_B:.2f}")
    print(f"Mean Absolute Error: {mae_A_on_B:.2f}")
    print(f"R^2 Score: {r2_A_on_B:.2f}")

    print("==========result of A on C==========")
    print(f"Mean Squared Error: {mse_A_on_C:.2f}")
    print(f"Mean Absolute Error: {mae_A_on_C:.2f}")
    print(f"R^2 Score: {r2_A_on_C:.2f}")

    print("==========result of B on A==========")
    print(f"Mean Squared Error: {mse_B_on_A:.2f}")
    print(f"Mean Absolute Error: {mae_B_on_A:.2f}")
    print(f"R^2 Score: {r2_B_on_A:.2f}")

    print("==========result of B on B==========")
    print(f"Mean Squared Error: {mse_B_on_B:.2f}")
    print(f"Mean Absolute Error: {mae_B_on_B:.2f}")
    print(f"R^2 Score: {r2_B_on_B:.2f}")

    print("==========result of B on C==========")
    print(f"Mean Squared Error: {mse_B_on_C:.2f}")
    print(f"Mean Absolute Error: {mae_B_on_C:.2f}")
    print(f"R^2 Score: {r2_B_on_C:.2f}")

    print("==========result of C on A==========")
    print(f"Mean Squared Error: {mse_C_on_A:.2f}")
    print(f"Mean Absolute Error: {mae_C_on_A:.2f}")
    print(f"R^2 Score: {r2_C_on_A:.2f}")

    print("==========result of C on B==========")
    print(f"Mean Squared Error: {mse_C_on_B:.2f}")
    print(f"Mean Absolute Error: {mae_C_on_B:.2f}")
    print(f"R^2 Score: {r2_C_on_B:.2f}")

    print("==========result of C on C==========")
    print(f"Mean Squared Error: {mse_C_on_C:.2f}")
    print(f"Mean Absolute Error: {mae_C_on_C:.2f}")
    print(f"R^2 Score: {r2_C_on_C:.2f}")