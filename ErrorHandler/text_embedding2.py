from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

class BERTEmbeddingTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', pooling='masked_mean', 
                 max_length=256, batch_size=64, dynamic_padding=True,
                 device=None, verbose=True, truncation_strategy='smart_head'):
        """
        Optimized BERT text embedding transformer for scikit-learn pipelines.

        Parameters:
        - model_name (str): HuggingFace model name/path
        - pooling (str): ['cls', 'mean', 'max', 'masked_mean'], masked_mean is the best theoretically
        - max_length (int): Max sequence length (applied before dynamic padding)
        - batch_size (int): Adaptive batch size (auto-reduce on OOM)
        - dynamic_padding (bool): Enable smart padding per batch
        - device (str): Override auto device detection
        - verbose (bool): Enable progress logging
        - truncation_strategy (str): ['smart_head', 'head_tail', 'head'], smart truncation strategy
        """
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.dynamic_padding = dynamic_padding
        self.verbose = verbose
        self.truncation_strategy = truncation_strategy
        
        # Device configuration
        self.device = self._detect_device(device)
        
        # Lazy initialization
        self.tokenizer = None
        self.model = None
        self.data_collator = None

    def _detect_device(self, device):
        """Auto-configure compute device"""
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _initialize_components(self):
        """Lazy initialization of heavy components"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.actual_max_length = min(self.max_length, self.tokenizer.model_max_length)
            
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Disable dropout
            
        if self.data_collator is None and self.dynamic_padding:
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='longest',
                max_length=min(self.max_length, self.tokenizer.model_max_length)
            )

    def _smart_truncate(self, text):
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= self.actual_max_length:
            return text

        if self.truncation_strategy == 'smart_head':
            # 保留包含关键信号的开头部分
            important_keywords = ['but', 'however', 'although', 'except']
            sentences = text.split('.')
            truncated = []
            for sent in sentences:
                if any(kw in sent.lower() for kw in important_keywords):
                    break  # 遇到转折词停止截断
                truncated.append(sent)
                if len(self.tokenizer.tokenize('.'.join(truncated))) > self.actual_max_length:
                    truncated.pop()  # 移除最后一句
                    break
            return '.'.join(truncated).strip()

        elif self.truncation_strategy == 'head_tail':
            # 首尾组合截断
            head_len = self.actual_max_length // 2
            tail_len = self.actual_max_length - head_len
            head = tokens[:head_len]
            tail = tokens[-tail_len:]
            return self.tokenizer.convert_tokens_to_string(head + tail)

        else:
            # 默认头部截断
            return self.tokenizer.convert_tokens_to_string(tokens[:self.actual_max_length])
        
        
    def _smart_batching(self, texts):
        """Sort texts by length for efficient padding"""
        tokenized = [self.tokenizer.tokenize(text) for text in texts]
        indices = sorted(range(len(tokenized)), 
                        key=lambda i: len(tokenized[i]), 
                        reverse=True)
        return [texts[i] for i in indices]

    def _pool_embeddings(self, hidden_states, attention_mask):
        """Enhanced pooling strategies with mask support"""
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]
            
        elif self.pooling == 'masked_mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.pooling == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding to large negative value
            return torch.max(hidden_states, 1)[0]
            
        elif self.pooling == 'mean':
            return torch.mean(hidden_states, dim=1)
            
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def fit(self, X, y=None):
        """Validation and component initialization"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
            
        if X.shape[1] != 1:
            raise ValueError("DataFrame must contain exactly one text column")
            
        self._initialize_components()
        return self

    def transform(self, X):
        """Main transformation logic"""
        self._initialize_components()
        
        text_column = X.columns[0]
        texts = X[text_column].astype(str).tolist()
        
        # 应用智能截断预处理
        processed_texts = [
            self._smart_truncate(text) 
            if len(self.tokenizer.tokenize(text)) > self.actual_max_length 
            else text 
            for text in tqdm(texts, desc="Truncating texts", disable=not self.verbose)
        ]
        
        # Sort texts for efficient batching
        if self.dynamic_padding:
            processed_texts = self._smart_batching(processed_texts)
            
        embeddings = []
        current_batch_size = self.batch_size
        
        with torch.inference_mode():
            for i in tqdm(range(0, len(processed_texts), current_batch_size),
                        desc="Processing batches",
                        disable=not self.verbose):
                batch_processed_texts = processed_texts[i:i + current_batch_size]
                
                try:
                    # Dynamic tokenization
                    if self.dynamic_padding:
                        encoded = [self.tokenizer.encode(text, truncation=True) for text in batch_processed_texts]
                        inputs = self.data_collator([{"input_ids": ids} for ids in encoded])
                    else:
                        inputs = self.tokenizer(
                            batch_processed_texts,
                            return_tensors="pt",
                            truncation=True,
                            padding="max_length",
                            max_length=self.max_length
                        )
                        
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs)
                    batch_embeddings = self._pool_embeddings(
                        outputs.last_hidden_state,
                        inputs.get('attention_mask', None)
                    )
                    
                    embeddings.append(batch_embeddings.cpu().numpy())
                    
                except RuntimeError as e:  # Handle OOM errors
                    if 'CUDA out of memory' in str(e) and current_batch_size > 1:
                        warnings.warn(f"OOM detected, reducing batch size from {current_batch_size} to {current_batch_size//2}")
                        current_batch_size = max(current_batch_size // 2, 1)
                        continue
                    else:
                        raise

        # Preserve original order
        if self.dynamic_padding:
            reverse_indices = np.argsort(np.arange(len(processed_texts)))
            embeddings = np.concatenate(embeddings)[reverse_indices]
        else:
            embeddings = np.concatenate(embeddings)
            
        return pd.DataFrame(embeddings, index=X.index)

    def __getstate__(self):
        """Handle serialization for scikit-learn/pickling"""
        state = self.__dict__.copy()
        # Remove un-pickleable components
        state['model'] = None
        state['tokenizer'] = None
        state['data_collator'] = None
        return state

if __name__ == "__main__":
    # Enhanced test case
    sample_texts = [
        "This is a positive review! Loved the product quality.",
        "Terrible experience. Would not recommend.",
        "Neutral opinion. The item works but packaging was damaged.",
        "Absolutely fantastic! Exceeded all expectations."
    ] * 250  # Scale test data
    
    df = pd.DataFrame({"review/text": sample_texts})
    
    # Test configurations
    configs = [
        {"pooling": "cls", "dynamic_padding": False},
        {"pooling": "masked_mean", "dynamic_padding": True},
        {"batch_size": 128, "max_length": 128}
    ]
    
    for cfg in configs:
        print(f"\nTesting config: {cfg}")
        transformer = BERTEmbeddingTransformer2(**cfg, verbose=True)
        
        import time
        start = time.time()
        embeddings = transformer.transform(df)
        elapsed = time.time() - start
        
        print(f"Shape: {embeddings.shape}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Sample embedding:\n{embeddings.iloc[0].values[:10]}")  # Show partial embedding