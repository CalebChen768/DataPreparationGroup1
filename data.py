from datasets import load_dataset
import pandas as pd
from errors import add_missing_values, add_outliers, add_typo_to_text_column, add_gibberish_to_text_column
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
    df['beer/ABV'] = df['beer/ABV'].astype(float)
    
    df['beer/style'] = df['beer/style'].apply(html.unescape)
    df['beer/name'] = df['beer/name'].apply(html.unescape)
    df['review/time'] = pd.to_datetime(df['review/time'], unit='s')
    

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
                    sampling=error['sampling'],
                ).transform(with_errors)
            elif error['type'] == 'typo':
                with_errors = add_typo_to_text_column(
                    with_errors, 
                    error['column'], 
                    error['typo_rate']
                )
            elif error['type'] == 'gibberish':
                with_errors = add_gibberish_to_text_column(
                    with_errors,
                    error['column'],
                    error['gibberish_rate']
                )
        return with_errors

if __name__ == '__main__':
    # df = load_ratebeer()

    # #with_errors = df.iloc[0:10, ]
    # df = df.head(100000)
    # df.to_csv("data_10w_002001.csv", index = False)

    # #head = df.head(100000)

    # error_adder = DataErrorAdder(config_path='config.yaml')
    # with_errors = error_adder.add_errors(df)
    # #head_errors = error_adder.add_errors(head)
    # #with_errors.to_csv("error2_1w.csv", index = False)

    # print(with_errors)
    # with_errors.to_csv("error_10W_002001.csv", index = False)
    # #print(with_errors["text"])


    #from sklearn.datasets import fetch_california_housing
    #california = fetch_california_housing()

    # 将特征数据转换为 DataFrame
    #df = pd.DataFrame(california.data, columns=california.feature_names)

    #df["MedHouseVal"] = california.target

    df = pd.read_csv('games_p_63660.csv')
    df[['rating_number', 'helpful_vote','rating']] = df[['rating_number','helpful_vote','rating']].astype(float)
    df['price'] = df['price'].str.extract('(\d+\.?\d*)', expand=False).astype(float)

    #df = df.head(15000)

    counts = df['price'].apply(type).value_counts()
    print(counts)


    # 保存到当前目录（默认路径）
    #df.to_csv("game_A_1005.csv", index=False)
    error_adder = DataErrorAdder(config_path='game.yaml')
    with_errors = error_adder.add_errors(df)
    #with_errors.to_csv("game_1005.csv", index = False)
    print(df)