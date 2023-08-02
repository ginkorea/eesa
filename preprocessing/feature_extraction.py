from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


class FeatureExtractor:

    def __init__(self, dataset):
        self.dataset = dataset
        self.vectorizer = None
        self.vector = None

    def __repr__(self):
        return self.vector

    def fit_vectorizer(self):
        self.vectorizer.fit(self.dataset)

    def text2vector(self, text_sample):
        return self.vectorizer.transform([text_sample])

    def process(self, verbose=True):
        for text in self.dataset:
            vector = self.text2vector(text)
            if verbose:
                print("Text:", text)
                print("Vector:", vector.toarray())
                print()


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

    def __init__(self, data):
        self.vector = []
        self.vector_list = []
        self.data = data
        self.bow = BagOfWords(data)
        self.tf_idf = TermFreqInverseDocFreq(data)
        self.ngram = NGram(data)
        self.extractors = [self.bow, self.tf_idf, self.ngram]

    def fit(self):
        for extractor in self.extractors:
            extractor.fit_vectorizer()

    def extract(self, text, verbose=False):
        vector = []
        parsed = []
        for extractor in self.extractors:
            sub_vector = extractor.text2vector(text)
            vector.append(sub_vector.toarray())
        for array in vector:
            array_to_list = list(array[0])
            for number in array_to_list:
                parsed.append(number)
        parsed_array = np.array(parsed)
        if verbose:
            print("parsed: %s" % parsed_array)
        return parsed_array

    def process(self):
        for text in self.data:
            vector = self.extract(text)
            self.vector_list.append(vector)

    def merge(self):
        for extractor in self.extractors:
            array = extractor.vector.toarray()
            array_to_list = list(array[0])
            for number in array_to_list:
                self.vector.append(number)
        print(self.vector)

    def vec2array(self):
        return np.array(self.vector, dtype="float")


def test_fe():
    texts = [
        "I love running in the mornings.",
        "Runners are running away while running shoes are running.",
        "I love running and swimming for fitness.",
    ]

    multi = MultiExtractor(texts)
    multi.fit()
    multi.process()
    print(len(multi.vector_list))



