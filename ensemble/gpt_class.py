import numpy as np

from llm.sentiment import *
from data.data import *
import pandas as pd
import re

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


def label_row(text, numb_classifier=3, verbose=False):
    def float_conversion(input_string):
        # Regular expression to extract numbers from the string
        # cyan(input_string)
        try:
            number_match = re.findall(r'-?\d+\.\d+|-?\d+', input_string)
            if number_match:
                return float(number_match[0])
            else:
                return np.NaN
        except TypeError:
            return input_string

    def clean_columns(parsed_text):
        for i, result in enumerate(parsed_text):
            j = 0
            while j < 3:
                parsed_text[i][j] = float_conversion(parsed_text[i][j])
                j += 1
        return parsed_text

    gpt_class = LLMClassifier(numb_of_classifiers=numb_classifier, verbose=verbose)
    gpt_class.classify(text)
    parsed_results = gpt_class.parse_results()
    parsed_results = clean_columns(parsed_results)
    column_names = ["sentiment_score", "confidence_rating", "explanation_score", "explanation"]
    try:
        parsed_df = pd.DataFrame(data=parsed_results, columns=column_names)
    except ValueError as e:
        red("error processing parsed_results: %s \n %s" % (e, parsed_results))
        exit()
    try:
        senti_score = parsed_df['sentiment_score'].mean()
    except TypeError as e:
        red("error: %s" % e)
        red(parsed_df['sentiment_score'])
        senti_score = None
    try:
        conf_rating = parsed_df['confidence_rating'].mean()
    except TypeError as e:
        red("error: %s" % e)
        red(parsed_df['confidence_rating'])
        conf_rating = None
    try:
        ex_score = parsed_df['explanation_score'].mean()
    except TypeError as e:
        red("error: %s" % e)
        red(parsed_df['explanation_score'])
        ex_score = None
    try:
        explanation_str = ' '.join(parsed_df['explanation'])
        senti_summarizer = SentiSummary()
        senti_summary = senti_summarizer.classify_sentiment(explanation_str, verbose=verbose)
    except TypeError as e:
        red("error: %s" % e)
        red('explanation: %s' % parsed_df['explanation'])
        senti_summary = parsed_df['explanation'][0]
    averaged_results = [senti_score, conf_rating, ex_score, senti_summary]
    return averaged_results


def test_label():
    test = "This is a test of the national broadcast system.  You are not dying."
    test_results = label_row(test)
    green("label: %s" % test_results)
