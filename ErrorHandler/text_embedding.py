from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np

class BERTEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', pooling='cls', max_length=128, batch_size=64, device=None):
        """
        A Scikit-learn compatible transformer for BERT embeddings.

        Parameters:
        - model_name (str): Pretrained BERT model name.
        - pooling (str): Pooling strategy, choose from:
            - "cls"  -> Use the [CLS] token representation.
            - "mean" -> Mean pooling of all token embeddings.
            - "max"  -> Max pooling of all token embeddings.
        - max_length (int): Maximum token length for truncation.
        - batch_size (int): Number of sentences processed in one batch.
        - device (str or None): Choose device:
            - "cuda" -> Use GPU (if available).
            - "mps"  -> Use Apple Metal GPU (for M1/M2/M3 Macs).
            - "cpu"  -> Use CPU.
            - None   -> Auto detect.
        """
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")

    def fit(self, X, y=None):
        """
        No training needed; just ensure input is a DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        return self  # Scikit-learn pipeline compatibility

    def transform(self, X):
        """
        Convert text data in a Pandas DataFrame column to BERT embeddings.

        Parameters:
        - X (pd.DataFrame): DataFrame containing a single text column.

        Returns:
        - pd.DataFrame: DataFrame containing BERT embeddings as numpy arrays.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Ensure there is at least one text column
        if X.shape[1] != 1:
            raise ValueError("Input DataFrame must contain exactly one text column")
        
        # Extract text column
        text_column = X.columns[0]
        texts = X[text_column].astype(str).tolist()

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        all_embeddings = []

        # Batch processing
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenization
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

            # Apply pooling
            if self.pooling == "cls":
                pooled = last_hidden_state[:, 0, :]  # CLS token
            elif self.pooling == "mean":
                pooled = last_hidden_state.mean(dim=1)  # Mean pooling
            elif self.pooling == "max":
                pooled, _ = last_hidden_state.max(dim=1)  # Max pooling
            else:
                raise ValueError("pooling must be 'cls', 'mean' or 'max'")

            # Convert to CPU numpy arrays
            all_embeddings.append(pooled.cpu().numpy())

        # Stack all embeddings into a single numpy array
        embeddings = np.vstack(all_embeddings)

        # Convert to Pandas DataFrame, keeping original index
        return pd.DataFrame(embeddings, index=X.index)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./sampled_dataframe.csv")
    df.dropna(inplace=True)
    bert_transformer = BERTEmbeddingTransformer()

    # Test on a sample DataFrame
    import time
    start = time.time()
    embeddings = bert_transformer.transform(df['review/text'].loc[:1000].to_frame())
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Embeddings shape: {embeddings.shape}")
    print(embeddings)