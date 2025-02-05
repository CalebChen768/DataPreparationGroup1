from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import string
import math
from collections import Counter
from gibberish_detector import detector

import pandas as pd
import fasttext
import string
import os
import requests
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import words
import nltk

class GibberishDetector(BaseEstimator, TransformerMixin):
    def __init__(self, method="entropy", threshold=2.5, custom_func=None, ngram_model_path=None):
        """
        Parameters:
        method (str): gibberish detection method:
                      - "entropy" (based on entropy calculation)
                      - "char_dist" (based on character distribution)
                      - "ngram" (based on gibberish-detector n-gram language model)
                      - "custom" (custom method)
        threshold (float): threshold (applicable for entropy and char_dist)
        custom_func (callable): custom method (must be provided when method="custom")
        ngram_model_path (str): path to the trained n-gram model (used when method="ngram")
        """
        self.method = method
        self.threshold = threshold
        self.custom_func = custom_func
        self.ngram_model_path = ngram_model_path

        if self.method not in ["entropy", "char_dist", "ngram", "custom"]:
            raise ValueError("method must be 'entropy', 'char_dist', 'ngram' or 'custom'")
        if self.method == "custom" and not callable(self.custom_func):
            raise ValueError("custom method requires custom_func to be provided")
        
        # Initialize gibberish detector (n-gram method)
        self.gibberish_detector = None
        if self.method == "ngram":
            self.gibberish_detector = detector.create_from_model('big.model')

            if self.ngram_model_path:
                self.gibberish_detector.load_model(self.ngram_model_path)  # Load the trained n-gram model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is DataFrame or Series
        if isinstance(X, pd.DataFrame):
            result = X.applymap(self._detect_gibberish)
        elif isinstance(X, pd.Series):
            result = X.apply(self._detect_gibberish)
        else:
            raise TypeError("Input must be pandas DataFrame or Series")
        # Convert True to NaN
        X = X.where(~result, np.nan)  # Keep False values, replace True with NaN
        # X.dropna(axis=0, inplace=True)  
        return pd.DataFrame(X)

    def _detect_gibberish(self, text):
        """Detect gibberish in a single text"""
        if not isinstance(text, str) or text.strip() == "":
            return False  # Empty or non-text values are not considered gibberish

        if self.method == "entropy":
            return self._entropy_based(text)
        elif self.method == "char_dist":
            return self._char_distribution_based(text)
        elif self.method == "ngram":
            return self._ngram_based(text)
        elif self.method == "custom":
            return self.custom_func(text)
        return False

    def _entropy_based(self, text):
        """
        Detect gibberish using Shannon entropy:
        - Gibberish has higher entropy (random character sequences have higher entropy)
        - Natural language has lower entropy (has patterns)
        """
        text = text.lower().translate(str.maketrans('', '', string.punctuation + string.digits))  # Keep only letters
        if len(text) == 0:
            return False  # No valid characters

        char_counts = Counter(text)
        total_chars = sum(char_counts.values())

        entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())

        # print(entropy < self.threshold)
        return entropy < self.threshold

    def _char_distribution_based(self, text):
        """
        Calculate the character distribution of the text to detect deviation from natural language:
        - Gibberish has more uniform character distribution
        - Natural language has less uniform character distribution
        """
        text = text.lower().translate(str.maketrans('', '', string.punctuation + string.digits))  # Keep only letters
        if len(text) == 0:
            return False

        char_counts = Counter(text)
        total_chars = sum(char_counts.values())
        char_frequencies = np.array([count / total_chars for count in char_counts.values()])

        # Calculate the standard deviation of character distribution (lower -> gibberish, higher -> natural language)
        std_dev = np.std(char_frequencies)
        return std_dev < self.threshold

    def _ngram_based(self, text):
        """
        Use gibberish-detector's n-gram method
        """
        if self.gibberish_detector is None:
            raise RuntimeError("n-gram model not loaded correctly, please check ngram_model_path")
        # print(self.gibberish_detector.is_gibberish(text))
        return self.gibberish_detector.is_gibberish(text)

    def get_feature_names_out(self, X):
        return X.columns


class TwoStepGibberishDetector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.62, max_candidates=5):
        nltk.download('words')
        # self.columns = columns
        
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.model_path = 'lid.176.bin'
        self._download_model()
        self.model_d = fasttext.load_model(self.model_path)
        self.english_vocab = set(words.words())

    def _download_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model '{self.model_path}' not found. Downloading...")
            url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
            response = requests.get(url, stream=True)
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Model '{self.model_path}' downloaded successfully!")
        else:
            print(f"Model '{self.model_path}' already exists.")

    def _clean_text(self, text):
        
        text = text.lower()  
        text = text.translate(str.maketrans('', '', string.punctuation)) 

        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X.dropna(axis=0, inplace=True)
        self.columns = X.columns
        row_count = X.shape[0]
        for column in self.columns:
            not_en_index = []
            col_index = X.columns.get_loc(column)
            for i in range(0, row_count):
                if(pd.isnull(X.iloc[i, col_index])):
                    continue
                # print("processing row:", i)
                target = self._clean_text(X.iloc[i, col_index])
            
                predictions = self.model_d.predict(target)
                language = predictions[0][0].split('__label__')[1]
                confidence = predictions[1][0]
                # print("Confidence:", confidence)
                if confidence < self.threshold:
                    # if(i/100==0):
                    #     print("Processing row:", i)
                    #     print(target)
                    tokens = target.split()
                    if len(tokens) < 7:
                        word_count = 0
                        for token in tokens:
                            token = token.translate(str.maketrans('', '', string.punctuation))
                            if token.lower() in self.english_vocab:
                                # print("Go to set compare")
                                word_count += 1
                            if word_count > len(tokens) - word_count:
                                language = "en"
                            elif language == "en":
                                language = "nonsense"
                    elif language == "en":
                        language = "nonsense"

                if language != "en":
                    not_en_index.append(i)
            X.loc[not_en_index, column] = np.nan
            
        X.dropna(axis=0, inplace=True)

        return X




if __name__ == "__main__":
    df = pd.DataFrame({
        "text": ["dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq", "22 madhur old punjab pickle chennai", "I love this website", "Madhur study in a teacher"]
    })

    # print(df.shape)
    from sklearn.compose import ColumnTransformer

    trans = ColumnTransformer([
        ("gibberish", GibberishDetector(method="entropy", threshold=3.5), "text")
    ])

    # print(trans.fit_transform(df))