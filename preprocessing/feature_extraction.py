from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


class FeatureExtractor:
    def __init__(self, dataset: list[str]):
        self.dataset = dataset
        self.vectorizer = None

    def fit_vectorizer(self):
        self.vectorizer.fit(self.dataset)

    def text2vector(self, text: str):
        return self.vectorizer.transform([text])


class BagOfWords(FeatureExtractor):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.vectorizer = CountVectorizer()


class TermFreqInverseDocFreq(FeatureExtractor):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.vectorizer = TfidfVectorizer()


class NGram(FeatureExtractor):
    def __init__(self, dataset, ngram_range=(2, 2)):
        super().__init__(dataset)
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)


class MultiExtractor:
    def __init__(self, data: list[str]):
        self.data = data
        self.vector_list = []
        self.extractors = [
            BagOfWords(data),
            TermFreqInverseDocFreq(data),
            NGram(data)
        ]

    def fit(self):
        for extractor in self.extractors:
            extractor.fit_vectorizer()

    def extract(self, text: str, verbose: bool = False) -> np.ndarray:
        vectors = []
        for extractor in self.extractors:
            vec = extractor.text2vector(text).toarray()[0]
            vectors.extend(vec)
        output = np.array(vectors)
        if verbose:
            print(f"[Vector]: {output}")
        return output

    def process(self):
        for text in self.data:
            self.vector_list.append(self.extract(text))

    def vec2array(self):
        return np.array(self.vector_list, dtype="float")


### tests

def test_feature_extraction():
    # Sample dataset
    dataset = [
        "This is a positive review.",
        "I did not like this product.",
        "The service was excellent!",
        "I am not satisfied with the quality."
    ]

    # Initialize MultiExtractor
    extractor = MultiExtractor(dataset)
    extractor.fit()
    extractor.process()

    # Check the output
    print(extractor.vec2array())

if __name__ == "__main__":
    test_feature_extraction()