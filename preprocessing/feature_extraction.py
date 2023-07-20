from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
        self.vector = self.vectorizer.transform([text_sample])

    def process(self):
        self.fit_vectorizer()
        for text in self.dataset:
            self.text2vector(text)
            print("Text:", text)
            print("Vector:", self.vector.toarray())
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


texts = [
    "I love running in the mornings.",
    "Runners are running away while running shoes are running.",
    "I love running and swimming for fitness.",
]

bow = BagOfWords(texts)
bow.process()

tf_idf = TermFreqInverseDocFreq(texts)
tf_idf.process()

ngram = NGram(texts)
ngram.process()