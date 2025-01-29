# python -m spacy download en_core_web_md     (or en_core_web_sm)

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
from spellchecker import SpellChecker
import torch

class GrammarCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 model_name="vennify/t5-base-grammar-correction",
                 spell_check=True,
                 num_beams=5,
                 max_length=256,
                 device="auto"):
        
        self.model_name = model_name
        self.spell_check = spell_check
        self.num_beams = num_beams
        self.max_length = max_length
        self.device = device
        
        self.grammar_model = None
        self.tokenizer = None
        self.spell = None
        self.nlp = None
        self.sent_nlp = None

    def _init_components(self):
        """Initialization with delay"""
        if self.grammar_model is None:
            self.grammar_model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if "cuda" in str(self.device) else None
            )
            
        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
        if self.spell_check and self.spell is None:
            self.spell = SpellChecker()
            
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_md")
            self.sent_nlp = spacy.load("en_core_web_md", disable=["parser", "tagger", "ner"])

    def fit(self, X, y=None):
        self._init_components()
        return self

    def transform(self, X):
        self._init_components()
        
        if isinstance(X, str):
            return self._correct_single(X)
            
        return [self._correct_single(text) for text in X]

    def _correct_single(self, text):
        try:
            if self.spell_check:
                text = self._correct_spelling(text)
            
            return self._correct_grammar(text)
        except Exception as e:
            print(f"Error processing text: {text[:50]}... - {str(e)}")
            return text # Return original text

    def _correct_spelling(self, text):
        words = [token.text for token in self.sent_nlp(text)]
        corrected = [
            self.spell.correction(word) if self.spell.correction(word) is not None else word
            for word in words
        ]
        return ' '.join(corrected)

    def _correct_grammar(self, text):
        inputs = self.tokenizer(
            f"grammar: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.grammar_model.device)

        outputs = self.grammar_model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('grammar_corrector', GrammarCorrector(
            spell_check=True,
            num_beams=3,
            device="auto"))
    ])

    test_texts = [
        "He go to school everyday.",
        "I has three apple in my bag.",
        "Their are many problem with this approach."
    ]

    corrected = pipeline.transform(test_texts)
    
    for orig, corr in zip(test_texts, corrected):
        print(f"Original: {orig}")
        print(f"Corrected: {corr}\n")