from datasets import load_dataset
import pandas as pd
from errors import add_missing_values, add_outliers, add_typo_to_text_column  
from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import Scaling
import html
import numpy as np
import yaml

def load_data():
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    return dataset["full"].to_pandas()

def load_ratebeer(path="./sampled_dataframe.csv"):
    df = pd.read_csv(path)

    # The following are some dataset specific prarsing
    df['review/appearance'] = df['review/appearance'].str.split('/').str[0]
    df['review/aroma'] = df['review/aroma'].str.split('/').str[0]
    df['review/palate'] = df['review/palate'].str.split('/').str[0]
    df['review/taste'] = df['review/taste'].str.split('/').str[0]
    df['review/overall'] = df['review/overall'].str.split('/').str[0]
    df['review/appearance'] = df['review/appearance'].astype(float)
    df['review/aroma'] = df['review/aroma'].astype(float)
    df['review/palate'] = df['review/palate'].astype(float)
    df['review/taste'] = df['review/taste'].astype(float)
    df['review/overall'] = df['review/overall'].astype(float)
    df['beer/ABV'] = df['beer/ABV'].replace("-", np.nan)
    df['beer/style'] = df['beer/style'].apply(html.unescape)
    df['beer/name'] = df['beer/name'].apply(html.unescape)
    df['review/time'] = pd.to_datetime(df['review/time'], unit='s')
    df['beer/ABV'] = df['beer/ABV'].astype(float)

    return df

class DataErrorAdder:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def add_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        with_errors = data.copy()
        for error in self.config['errors']:
            if error['type'] == 'missing_values':
                with_errors = MissingValues(
                    column=error['column'], 
                    fraction=error['fraction'], 
                    missingness=error['missingness']
                ).transform(with_errors)
            elif error['type'] == 'scaling':
                with_errors = Scaling(
                    column=error['column'], 
                    fraction=error['fraction'], 
                    sampling=error['sampling']
                ).transform(with_errors)
            elif error['type'] == 'typo':
                with_errors = add_typo_to_text_column(
                    with_errors, 
                    error['column'], 
                    error['typo_rate']
                )
        return with_errors

if __name__ == '__main__':
    df = load_data()

    with_errors = df.iloc[0:10, ]

    error_adder = DataErrorAdder(config_path='config.yaml')
    with_errors = error_adder.add_errors(with_errors)

    print(with_errors)
    print(with_errors["text"])
