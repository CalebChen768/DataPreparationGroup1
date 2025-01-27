from datasets import load_dataset
import pandas as pd
from errors import add_missing_values, add_outliers, add_typo_to_text_column  
from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import Scaling

def load_data():
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    return dataset["full"].to_pandas()

def add_errors(data):
    with_errors = data.copy()

    with_errors = MissingValues(column='verified_purchase', fraction=.4, missingness='MAR').transform(with_errors)
    with_errors = SwappedValues(column='title', fraction=.2, sampling='MAR').transform(with_errors)
    with_errors = Scaling(column='rating', fraction=.2, sampling='MAR').transform(with_errors)
    with_errors = add_typo_to_text_column(with_errors, "text")

    return with_errors

if __name__ == '__main__':
    df = load_data()

    with_errors = df.iloc[0:10, ]

    with_errors = add_errors(with_errors)

    print(with_errors)
    print(with_errors["text"])
