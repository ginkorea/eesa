# components.py
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.language_processing import process_text, set_up_stop_words
from preprocessing.feature_extraction import MultiExtractor
from ensemble.llm_classifier import label_row
import numpy as np


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Cleans, tokenizes, stems, and removes stopwords."""
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or set_up_stop_words()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join(process_text(text, self.stop_words)) for text in X]


class FeatureVectorizer(BaseEstimator, TransformerMixin):
    """Converts cleaned text into combined BoW, TF-IDF, and N-gram vectors."""
    def __init__(self):
        self.extractor = None

    def fit(self, X, y=None):
        self.extractor = MultiExtractor(X)
        self.extractor.fit()
        return self

    def transform(self, X):
        self.extractor.data = X
        self.extractor.process()
        return self.extractor.vec2array()


class LLMLabeler(BaseEstimator, TransformerMixin):
    """Calls OpenAI LLM to extract sentiment-related features."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = [label_row(text) for text in X]
        return np.array([r[:3] for r in results], dtype=float)  # Only score/confidence/explanation
