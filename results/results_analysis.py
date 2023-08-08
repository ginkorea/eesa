import pandas as pd
import numpy as np
from llm.util import *


class Metric:

    def __init__(self, file):
        self.data = pd.read_csv(file, sep="|")
        self.results_to_binary()
        self.true = self.data['sentiment']
        self.label = self.data['results']
        self.p = self.precision()
        self.a = self.accuracy()
        self.r = self.recall()

    def results_to_binary(self):
        self.data['results'] = self.data['results'].apply(lambda x: 1 if x > 0.5 else 0)

    def correct_predictions(self):
        correct_predictions = np.sum(np.array(self.true) == np.array(self.label))
        return correct_predictions

    def total_predictions(self):
        return len(self.label)

    def true_positive(self):
        true_pos = np.sum((np.array(self.true) == 1) & (np.array(self.label) == 1))
        return true_pos

    def pred_positive(self):
        pred_pos = np.sum(np.array(self.label) == 1)
        return pred_pos

    def actual_positive(self):
        act_pos = np.sum(np.array(self.true) == 1)
        return act_pos

    def accuracy(self):
        accuracy = self.correct_predictions() / self.total_predictions()
        return accuracy

    def precision(self):
        precision = self.true_positive() / self.pred_positive()
        return precision

    def recall(self):
        recall = self.true_positive() / self.actual_positive()
        return recall

    def results(self):
        green("Precision: %s\nAccuracy: %s\nRecall: %s" % (self.p, self.a, self.r))
        return [self.p, self.a, self.r]


def test_gold():
    m = Metric('gold_with_results.csv')
    n = Metric('gold_with_llm_results_with_results.csv')
    o = Metric('depth_6_gold_with_results.csv')
    p = Metric('depth_6_gold_with_llm_results_with_results.csv')
    results_1 = m.results()
    results_2 = n.results()
    results_3 = o.results()
    results_4 = p.results()


m = Metric('depth_3_movies_1000_with_results.csv')
n = Metric('depth_3_movies_1000_with_llm_results_with_results.csv')
results_m = m.results()
results_n = n.results()