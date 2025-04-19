import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.feature_extraction import MultiExtractor
from preprocessing.language_processing import process_text
from ensemble.llm_classifier import label_row

DEFAULT_STOP_WORDS = {"the", "is", "and", "a", "an", "it", "of", "to"}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Cleans and tokenizes input text, removing stop words and applying normalization.
    """

    def __init__(self, stop_words=None):
        self.stop_words = stop_words or DEFAULT_STOP_WORDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join(process_text(text, self.stop_words)) for text in X]


class FeatureVectorizer(BaseEstimator, TransformerMixin):
    """
    Extracts numerical feature vectors from cleaned text using TF-IDF + custom features.
    """

    def __init__(self):
        self.extractor = None

    def fit(self, X, y=None):
        self.extractor = MultiExtractor(X)
        self.extractor.fit()
        return self

    def transform(self, X):
        if self.extractor is None:
            raise ValueError("[FeatureVectorizer] Called transform before fit.")

        # Always use the fitted extractor (it has fitted vectorizers)
        extractor = self.extractor
        extractor.data = X
        extractor.vector_list = []  # reset in case reused
        extractor.process()
        result = extractor.vec2array()

        assert result.shape[0] == len(X), "[FeatureVectorizer] Output shape mismatch"
        return result


class LLMLabeler(BaseEstimator, TransformerMixin):
    """
    Uses GPT/LLM to label sentiment attributes (score, confidence, explanation score).
    Designed for training-time augmentation only.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._cache = {}

    def fit(self, X, y=None):
        self._cache = {
            i: label_row(text, verbose=self.verbose)[:3] for i, text in enumerate(X)
        }
        return self

    def transform(self, X):
        transformed = []
        for i, text in enumerate(X):
            if i in self._cache:
                transformed.append(self._cache[i])
            else:
                transformed.append(label_row(text, verbose=self.verbose)[:3])
        return np.array(transformed)
