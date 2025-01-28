import random
from sklearn.utils import shuffle
import numpy as np

def flip_labels(y, flip_rate=0.1):
    y_flipped = y.copy()
    n_flips = int(len(y) * flip_rate)
    indices = np.random.choice(len(y), size=n_flips, replace=False)
    classes = np.unique(y)
    for idx in indices:
        current_class = y_flipped[idx]
        other_classes = [cls for cls in classes if cls != current_class]
        y_flipped[idx] = np.random.choice(other_classes)
    return y_flipped


def add_typo_to_text(text, typo_rate=0.1):
    text_with_typos = []
    for word in text.split():
        if random.random() < typo_rate:
            char_list = list(word)
            if len(char_list) > 1:
                i = random.randint(0, len(char_list) - 1)
                # filter out special characters
                if (char_list[i] in [',', '.', '!', '?', '-', ':', ';', '(', ')', '"', "'"]):
                    pass
                else:
                    char_list[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
            word = ''.join(char_list)
        text_with_typos.append(word)
    return ' '.join(text_with_typos)

def add_typo_to_text_column(X, column_name, typo_rate=0.1):
    X_noisy = X.copy()
    X_noisy[column_name] = X_noisy[column_name].apply(lambda x: add_typo_to_text(x, typo_rate) if isinstance(x, str) else x)
    return X_noisy

def add_missing_values(X, missing_rate=0.1):
    X_with_missing = X.copy()
    mask = np.random.rand(*X_with_missing.shape) < missing_rate
    X_with_missing[mask] = np.nan
    return X_with_missing

def add_outliers(X, outlier_rate=0.05, magnitude=3):
    X_outliers = X.copy()
    for col in X_outliers.select_dtypes(include=["number"]).columns:
        n_outliers = int(outlier_rate * len(X_outliers))
        indices = np.random.choice(len(X_outliers), size=n_outliers, replace=False)
        outliers = magnitude * X_outliers[col].std() * np.random.choice([-1, 1], size=n_outliers)
        X_outliers.loc[indices, col] += outliers
    return X_outliers
