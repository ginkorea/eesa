from llm.sentiment import *
from data.data import *
import pandas as pd

openai.api_key = get_openai_key()


class LLMClassifier:

    def __init__(self, numb_of_classifiers=3, verbose=False):
        self.verbose = verbose
        self.classifiers = []
        for i in range(0, numb_of_classifiers):
            classifier = SentiChat()
            self.classifiers.append(classifier)
        # print(self.classifiers)
        self.results = []
        self.df = None

    def classify(self, text):
        for classifier in self.classifiers:
            result = classifier.classify_sentiment(text, verbose=self.verbose)
            self.results.append(result)

    def parse_results(self):
        parsed_results = []
        for result in self.results:
            parsed_result = result.split('|')
            parsed_results.append(parsed_result)
        for i, m in enumerate(parsed_results):
            for j, n in enumerate(m):
                try:
                    parsed_results[i][j] = float(n)
                except ValueError:
                    pass
        return parsed_results


def label_row(text, numb_classifier=5, verbose=False):
    gpt_class = LLMClassifier(numb_of_classifiers=numb_classifier, verbose=verbose)
    gpt_class.classify(text)
    parsed_results = gpt_class.parse_results()
    column_names = ["sentiment_score", "confidence_rating", "explanation_score", "explanation"]
    parsed_df = pd.DataFrame(data=parsed_results, columns=column_names)
    senti_score = parsed_df['sentiment_score'].mean()
    conf_rating = parsed_df['confidence_rating'].mean()
    ex_score = parsed_df['explanation_score'].mean()
    explanation_str = ' '.join(parsed_df['explanation'])
    senti_summarizer = SentiSummary()
    senti_summary = senti_summarizer.classify_sentiment(explanation_str, verbose=verbose)
    averaged_results = [senti_score, conf_rating, ex_score, senti_summary]
    return averaged_results


def test_label():
    test = "This is a test of the national broadcast system.  You are not dying."
    test_results = label_row(test)
    green("label: %s" % test_results)
