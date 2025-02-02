from sklearn.base import BaseEstimator, TransformerMixin
from data import load_data, add_errors
from errors import add_typo_to_text_column
from textblob import TextBlob
# from spellchecker import SpellChecker

class TypoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns=None):
        self.target_columns = target_columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cleaned = X.copy()
        for column in self.target_columns:
            cleaned[column] = self._clean(X[column])
        return cleaned

    def _clean(self, texts):
        # checker = SpellChecker()
        cleaned_texts = []

        for text in texts:
            cleaned_words = []
            words = text.split()
            for word in words:
                if len(word) > 3:
                    b = TextBlob(word.lower())
                    cleaned_words.append(str(b.correct()))
                else:
                    cleaned_words.append(word.lower())

            cleaned_texts.append(" ".join(cleaned_words))
            

        return cleaned_texts



if __name__ == '__main__':
    df = load_data()

    df = df.iloc[0:10, ]

    dftypos = add_typo_to_text_column(df, "text")

    print(dftypos['text'])

    corrected = TypoTransformer(target_columns=['text']).transform(dftypos)
    
    print(corrected['text'])
    