import pandas as pd
import fasttext
import string
import nltk
import random
import os
import requests

from nltk.corpus import words

# Change main to use different dataset for test.
# Directly run "python text_j.py" and you can see test results.
# 1. Controversial targets and their judgment results will be printed
# 2. The target judged as errors and the replacement results will be displayed.


# Import this file and call replace_error() to detect errors and make replacements for a Dataframe target. ("from text_j import replace_error")

# "columns"             should be a list of the columns you wish to detect errors and do replacement, however multi columns was not tested yet
# "threshold"           If you increase the value, it will be more strict and may identify normal text as error. (accuracy around 98% on "game_error.csv" with 0.62)
# "ref_columns"         columns used to search similar rows. make sure "rating" is the first element of this list.
# "max_candidates"      the maximun number of similar rows to select, will randomly choose 1 from the selected rows for replacement(interpolation).


# fasttext and model 'lid.176.bin'      Determine the probability that a sentence is a certain language
# nltk and model 'words'                Determine weather the target is an English word


nltk.download('words')
english_vocab = set(words.words())



def clean_text(text):
    
    text = text.lower()  
    text = text.translate(str.maketrans('', '', string.punctuation)) 

    return text

def replace_error(df,columns = ["text"], threshold=0.62, ref_columns = ["rating","parent_asin"], max_candidates = 5):


    def download_fasttext_model(url, model_path):
        response = requests.get(url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Model '{model_path}' downloaded successfully!")

    model_path = 'lid.176.bin'

    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found. Downloading...")
        download_fasttext_model('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', model_path)
    else:
        print(f"Model '{model_path}' already exists.")



    model_d = fasttext.load_model(model_path)

    row_count = df.shape[0]


    for column in columns:
        not_en_index = []
        col_index = df.columns.get_loc(column)
        for i in range(0, row_count):
            target = clean_text(df.iloc[i, col_index])

            predictions = model_d.predict(target)
            language = predictions[0][0].split('__label__')[1]

            language = predictions[0][0].split('__label__')[1]
            confidence = predictions[1][0]  
            

            # checking confidence
            if confidence < threshold:
                tokens = target.split()

                # short text exception
                if len(tokens) < 7:
                    word_count = 0
                    for token in tokens:
                        # remove Punctuation for english_vocab check
                        token = token.translate(str.maketrans('', '', string.punctuation)) 

                        if token.lower() in english_vocab:
                            word_count += 1
                        if word_count > len(tokens) - word_count:
                            language = "en"
                        elif language == "en":
                            language = "nonsense"
                elif language == "en":
                    language = "nonsense"  

            if language != "en":
                not_en_index.append(i)
        
        # Interpolation
        for index in not_en_index:
            current_ref_values = df.loc[index, ref_columns]
            candidates_index = []

            previous_v = df.loc[index, column]

            for candidate_index, row in df.iterrows():
                if candidate_index not in not_en_index and all(current_ref_values[col] == row[col] for col in ref_columns):
                    candidates_index.append(candidate_index)

                if len(candidates_index) >= max_candidates:
                    break
            
            # In case no similar candidates, set the value based on the rating
            if len(candidates_index) == 0:
                rating_value = df.loc[index, ref_columns[0]]

                if 3 < rating_value <= 5:
                    df.loc[index, column] = "good"
                elif rating_value == 3:
                    df.loc[index, column] = "normal"
                elif 0 <= rating_value < 3:
                    df.loc[index, column] = "bad"
                else: 
                    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                    print ("~~~~~~~~~~~~~~   rating range  error  ~~~~~~~~~~~~~~~\n")
                    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            else:
                random_index = random.randint(0, len(candidates_index) - 1)
                df.loc[index,column] = df.loc[random_index,column]
    
    # print(f"error count: {len(not_en_index)}")

    return df












# ========= for test only , detection_test almost the same as replace_error =========


def detection_test(df,columns = ["text"], threshold=0.62, ref_columns = ["rating","parent_asin"], max_candidates = 5):

    def download_fasttext_model(url, model_path):
        response = requests.get(url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Model '{model_path}' downloaded successfully!")

    model_path = 'lid.176.bin'

    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found. Downloading...")
        download_fasttext_model('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', model_path)
    else:
        print(f"Model '{model_path}' already exists.")
        

    model_d = fasttext.load_model('lid.176.bin')

    row_count = df.shape[0]

    not_en_count = 0

    for column in columns:
        not_en_index = []
        col_index = df.columns.get_loc(column)
        for i in range(0, row_count):
            target = clean_text(df.iloc[i, col_index])

            predictions = model_d.predict(target)
            language = predictions[0][0].split('__label__')[1]

            language = predictions[0][0].split('__label__')[1]
            confidence = predictions[1][0]  
            

            # checking confidence
            if confidence < 0.8:
                if confidence < threshold:
                    tokens = target.split()

                    # short text exception
                    if len(tokens) < 7:
                        word_count = 0
                        for token in tokens:
                            token = token.translate(str.maketrans('', '', string.punctuation))
                            if token.lower() in english_vocab:
                                word_count += 1
                            else: print(token)
                            if word_count >= len(tokens) - word_count:
                                language = "en"
                            elif language == "en":
                                language = "nonsense"
                    elif language == "en":
                        language = "nonsense"  

                if language != "en":
                    not_en_count += 1
                    not_en_index.append(i)

                print(f"column: {column}, row: {i+2},language : {language}, confidence: {confidence}")
                print(f"target: {target}\n")

        # Interpolation

        print("=============  Interpolation  ============\n")

        counter = 0

        for index in not_en_index:
            current_ref_values = df.loc[index, ref_columns]
            candidates_index = []

            previous_v = df.loc[index, column]

            for candidate_index, row in df.iterrows():
                if candidate_index not in not_en_index and all(current_ref_values[col] == row[col] for col in ref_columns):
                    candidates_index.append(candidate_index)

                if len(candidates_index) >= max_candidates:
                    break
            
            # In case no similar candidates, set the value based on the rating
            if len(candidates_index) == 0:
                rating_value = df.loc[index, ref_columns[0]]

                if 3 < rating_value <= 5:
                    df.loc[index, column] = "good"
                elif rating_value == 3:
                    df.loc[index, column] = "normal"
                elif 0 <= rating_value < 3:
                    df.loc[index, column] = "bad"
                else: 
                    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                    print ("~~~~~~~~~~~~~~   rating range  error  ~~~~~~~~~~~~~~~\n")
                    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

                counter += 1
                print(f"{counter} no available candidate")
                print(f"replacing index({index + 2}) column({column})")
                print(f"from({previous_v}) to ({df.loc[index, column]})\n")

            else:
                random_index = random.randint(0, len(candidates_index) - 1)
                df.loc[index,column] = df.loc[random_index,column]

                counter += 1
                print(f"{counter} found {len(candidates_index)} candidates")
                print(f"replacing index({index + 2}) column({column})")
                print(f"from({previous_v}) to ({df.loc[index, column]})\n")



    #print(f"index count {len(not_en_index)}")
    print(f"error count: {not_en_count}")


if __name__ == "__main__":
    df = pd.read_csv('game_error.csv')
    #df = replace_error(df)
    #df.to_csv("game_error_corrected.csv",index=False)
    detection_test(df)