import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.feature_extraction import MultiExtractor
from preprocessing.language_processing import process_text
from ensemble.llm_classifier import label_row

# Optional fallback stop words
DEFAULT_STOP_WORDS = {"the", "is", "and", "a", "an", "it", "of", "to"}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or DEFAULT_STOP_WORDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for text in X:
            try:
                clean = process_text(text, _stop_words=self.stop_words)
                out.append(clean)
            except Exception as e:
                print(f"[TextPreprocessor Error] {text} → {e}")
                out.append("")
        return out


class FeatureVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.extractor = None

    def fit(self, X, y=None):
        self.extractor = MultiExtractor(X)
        self.extractor.fit()
        return self

    def transform(self, X):
        self.extractor.data = X
        self.extractor.process()
        result = self.extractor.vec2array()
        assert result.shape[0] == len(X), "[FeatureVectorizer] Output shape mismatch"
        return result


class LLMLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        vectors = []
        for idx, text in enumerate(X):
            try:
                result = label_row(text, verbose=self.verbose)
                if self.verbose:
                    print(f"[LLM {idx}] {result}")
                vectors.append(result[:3])
            except Exception as e:
                print(f"[LLMLabeler Error] {text}\n→ {e}")
                vectors.append([0.0, 0.0, 0.0])
        result_array = np.array(vectors, dtype=float)
        assert result_array.shape[0] == len(X), "[LLMLabeler] Output shape mismatch"
        return result_array


# ================================
# 🧪 COMPONENT TESTING
# ================================
if __name__ == "__main__":
    print("=== Running component tests ===\n")

    print("[1] Testing TextPreprocessor...")
    stop_words = {"the", "is", "and", "a", "an", "it", "of", "to"}
    test_data = [
        "This is a test sentence.",
        "Another example sentence with some stop words.",
        "Yet another one, just to be sure!"
    ]
    tp = TextPreprocessor(stop_words)
    tp_out = tp.transform(["I loved this movie!"])[0]
    print("→ Output:", tp_out)  # ✅ ADD THIS LINE
    assert isinstance(tp_out, list) and len(tp_out) >= 1 and all(isinstance(x, str) for x in tp_out)
    print("✓ TextPreprocessor passed.")


    print("\n[2] Testing FeatureVectorizer...")
    fv = FeatureVectorizer()
    fv_out = fv.fit_transform(test_data)
    assert isinstance(fv_out, np.ndarray)
    assert fv_out.shape[0] == 3
    print("✓ FeatureVectorizer passed.")

    print("\n[3] Testing LLMLabeler (verbose)...")
    llm = LLMLabeler(verbose=True)
    llm_out = llm.fit_transform(test_data)
    assert isinstance(llm_out, np.ndarray)
    assert llm_out.shape == (3, 3)
    print("✓ LLMLabeler passed.")

    print("\n=== All component tests completed successfully ===")


