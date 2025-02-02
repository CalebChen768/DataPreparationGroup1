import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from ErrorHandler import *
from data import load_ratebeer, DataErrorAdder

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.dummy import DummyClassifier, DummyRegressor

# # Create test data
# df = pd.DataFrame({
#     "col_1": [1, 2, 4, np.nan, 5, 6, -7, 8, 9, 1000, 5], 
#     "col_2": ["cat", "dog", None, "cat", "dog", "dog", "cat", "dog", "cat", "dog", "mice"],
#     "col_3": ["Y","N","Y","N","Y","N","Y","N","Y","N","Y"],
#     "col_4": [ "test I hey helo what calender insteresting",
#               "This game is interesting. I strongly recommend it.",
#               "This gmae is inteersting. I strongly recomend it.",
#               "I am a student",
#               "I am a student",
#               "My name is John",
#               "This is a test",
#               "What can I do for you",
#               "Haha, wonderful game",
#               "asdmko fndosfhjdn safijdpwhauih fidnafpii uhwafi nedwb",
#               "Do you like this game?"]
# })

# Create a pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", Pipeline([
#             ("missing_values", MissingValueChecker(data_type="numerical", strategy="drop")),
#             ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
#             ("outliers", OutlierHandler(strategy="clip")),
#             ("normalizer", ScaleAdjust(method="standard")),
#             ("align_index", AlignTransformer(original_index=df.index)),
#         ]), ["col_1"]),

#         ("cat", Pipeline([
#             ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
#             ("bound_constrain", OutOfBoundsChecker(allowed_set=["cat", "dog"])),
#             ("align_index", AlignTransformer(original_index=df.index)),
#             ("onehot",OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in ["col_2"]], handle_unknown="ignore"))
#         ]), ["col_2"]),

#         ("cat2", Pipeline([
#             ("onehot",OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in ["col_3"]], handle_unknown="ignore"))
#         ]), ["col_3"]),
        
#         ("text", Pipeline([
#             ("gibberish", GibberishDetector(method="ngram")),
#             ("align_index", AlignTransformer(original_index=df.index)),
#         ]), ["col_4"]),
#     ],
#     remainder="passthrough"
# )

# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X, columns=get_final_columns(preprocessor)), validate=False)),
#     ("dropper", MissDropper()),
# ])


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
            if isinstance(transformer, BERTEmbeddingTransformer2):
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
    # without manually add errors
    df0 = pd.read_csv("sampled_dataframe2.csv")
    # with manually add errors
    df = pd.read_csv("sampled_dataframe3.csv")

    df = df.head(15000)
    df0 = df0.head(15000)
    
    numerical_cols = ["review/appearance", "review/aroma", "review/palate", "review/taste"]
    target_col = "review/overall"
    categorical_cols = []
    text_cols = ["review/text"]

    # keep only the columns that are used
    df0 = df0[[target_col] + numerical_cols + categorical_cols + text_cols]
    df = df[[target_col] + numerical_cols + categorical_cols + text_cols]

    # pipeline with error detection
    preprocessor1 = ColumnTransformer(
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
                # ("gibberish", GibberishDetector(method="ngram")),
                ("gibberish", TwoStepGibberishDetector()),
                ("bert_embedding", BERTEmbeddingTransformer2()),
                ("align_index", AlignTransformer(original_index=df.index)),
                ("missing_text", MissingValueChecker(data_type="numerical", strategy="mean")),
                ("align_index2", AlignTransformer(original_index=df.index)),
            ]), text_cols),
        ],
        remainder="drop"
    )

    pipeline1 = Pipeline([
        ("preprocessor", preprocessor1),
        ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X, columns=get_final_columns(preprocessor1)))),
        ("dropper", MissDropper()),
    ])


    # pipeline without error detection
    preprocessor2 = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("missing_values", MissingValueChecker(data_type="numerical", strategy="drop")),
                # ("bound_constrain", OutOfBoundsChecker(lower_bound=0)),
                # ("outliers", OutlierHandler(strategy="clip")),
                ("normalizer", ScaleAdjust(method="standard")),
                ("align_index", AlignTransformer(original_index=df.index)),
            ]), numerical_cols),

            ("cat", Pipeline([
                ("missing_values", MissingValueChecker(data_type="categorical", strategy="drop")),
                ("align_index", AlignTransformer(original_index=df.index)),
                ("onehot", OneHotEncoder(categories=[list(set(df[i].unique())-{None, np.nan}) for i in categorical_cols], handle_unknown="ignore"))
            ]), categorical_cols),

            ("text", Pipeline([
                # ("gibberish", GibberishDetector(method="ngram")),
                ("bert_embedding", BERTEmbeddingTransformer2()),
                ("align_index", AlignTransformer(original_index=df.index)),
                ("missing_text", MissingValueChecker(data_type="numerical", strategy="drop")),
                ("align_index2", AlignTransformer(original_index=df.index)),
            ]), text_cols),
        ],
        remainder="drop"
    )

    pipeline2 = Pipeline([
        ("preprocessor", preprocessor2),
        ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X, columns=get_final_columns(preprocessor2)))),
        ("dropper", MissDropper()),
    ])



    # train a model
    np.random.seed(42)

    # print(df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X0 = df0.drop(columns=[target_col]) # A
    y0 = df0[target_col] # A
    
    # Split the dataset into training and testing sets
    pre_start = time.time()

    X1 = pipeline1.fit_transform(X)    
    # drop dropped rows from y
    y1 = y.loc[X1.index]
    pre_end = time.time()
    print(f"Time taken to preprocess pipeline with error detection: {pre_end - pre_start:.2f} seconds")
    print(f"Data shape: {X1.shape}")

    pre_start = time.time()
    X2 = pipeline2.fit_transform(X)
    y2 = y.loc[X2.index]
    pre_end = time.time()
    
    # X0 = pipeline2.fit_transform(X) # the pipeline drop some lines on X0
    # y0 = y.loc[X0.index]  # Align y0 with X0
    print(f"Time taken to preprocess pipeline without error detection: {pre_end - pre_start:.2f} seconds")
    print(f"Data shape: {X2.shape}")


    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2, random_state=42)

    start_fit = time.time()


    # Fit the model
    model1 = DecisionTreeRegressor(random_state=42)
    
    # fit the model with error detection
    model1.fit(X1_train, y1_train) # train on C
    end_fit = time.time()
    print(f"Time taken to fit: {end_fit - start_fit:.2f} seconds")

    # Test the model
    y1_pred = model1.predict(X1_test) # test on C

    # Dummy
    # dummy_reg1 = DummyRegressor(strategy="mean")
    # dummy_reg1.fit(X1_train, y1_train)

    # y1_pred_null_reg = dummy_reg1.predict(X1_test)

    # fit the model without error detection
    model2 = DecisionTreeRegressor(random_state=42)
    model2.fit(X2_train, y2_train) # train on B
    y2_pred = model2.predict(X1_test) # test on C

    # dummy_reg2 = DummyRegressor(strategy="mean")
    # dummy_reg2.fit(X2_train, y2_train)
    
    # y2_pred_null_reg = dummy_reg2.predict(X2_test)

    


    # dummy_cls = DummyClassifier(strategy="most_frequent")
    # dummy_cls.fit(X1_train, y1_train)

    # y_pred_null_cls = dummy_cls.predict(X1_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")

    # # draw a confusion matrix
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.show()

    # # print calss and index of their label
    # print(model.classes_)


    mse1 = mean_squared_error(y1_test, y1_pred)
    mae1 = mean_absolute_error(y1_test, y1_pred)
    r2_1 = r2_score(y1_test, y1_pred)


    # null_mse1 = mean_squared_error(y1_test, y1_pred_null_reg)
    # null_mae1 = mean_absolute_error(y1_test, y1_pred_null_reg)
    # null_r2_1 = r2_score(y1_test, y1_pred_null_reg)

    print("==========result in pipeline with error detection==========")
    print(f"Mean Squared Error: {mse1:.2f}")
    print(f"Mean Absolute Error: {mae1:.2f}")
    print(f"R^2 Score: {r2_1:.2f}")

    # print(f"Null Mean Squared Error: {null_mse1:.2f}")
    # print(f"Null Mean Absolute Error: {null_mae1:.2f}")
    # print(f"Null R^2 Score: {null_r2_1:.2f}")

    mse2 = mean_squared_error(y1_test, y2_pred)
    mae2 = mean_absolute_error(y1_test, y2_pred)
    r2_2 = r2_score(y1_test, y2_pred)

    # null_mse2 = mean_squared_error(y1_test, y2_pred_null_reg)
    # null_mae2 = mean_absolute_error(y1_test, y2_pred_null_reg)
    # null_r2_2 = r2_score(y1_test, y2_pred_null_reg)


    print("==========result in pipeline without error detection==========")
    print(f"Mean Squared Error: {mse2:.2f}")
    print(f"Mean Absolute Error: {mae2:.2f}")
    print(f"R^2 Score: {r2_2:.2f}")

    # print(f"Null Mean Squared Error: {null_mse2:.2f}")
    # print(f"Null Mean Absolute Error: {null_mae2:.2f}")
    # print(f"Null R^2 Score: {null_r2_2:.2f}")






    # print("============== Original Data ==============")
    # print(df)
    # print("\n============== Transformed Data ==============")
    # print(preprocessor.fit_transform(df))
    # print("\n============== Dropped NaN Data ==============")
    # print(pipeline.fit_transform(df))