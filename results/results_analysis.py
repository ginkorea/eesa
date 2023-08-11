import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from llm.util import *


class FoldParser:
    def __init__(
        self,
        file_to_parse,
        numb_folds=5,
        true="sentiment",
        label="results",
        confidence=0.95,
    ):
        self.data = pd.read_csv(file_to_parse, sep="|")
        self.name = file_to_parse
        self.folds = []
        self.parse(numb_folds=numb_folds)
        self.p_values = []
        self.a_values = []
        self.r_values = []
        self.par_values = []
        self.get_metrics(true=true, label=label)
        self.get_confidence_intervals(confidence=confidence)

    def parse(self, numb_folds=5):
        for i in range(0, numb_folds):
            this_fold = self.data[self.data["fold"] == i].copy()
            self.folds.append(this_fold)

    def get_metrics(self, true="sentiment", label="results"):
        for fold in self.folds:
            metric = Metric(fold, true=true, label=label)
            p, a, r = metric.results()
            self.p_values.append(p)
            self.a_values.append(a)
            self.r_values.append(r)
        self.p_values = np.array(self.p_values)
        self.a_values = np.array(self.a_values)
        self.r_values = np.array(self.r_values)
        self.par_values = [self.p_values, self.a_values, self.r_values]

    def get_confidence_intervals(self, confidence=0.95):
        for i, value in enumerate(self.par_values):
            ci = ConfidenceInterval(value, confidence=confidence)
            mean, margin_error = ci.get_ci()
            self.par_values[i] = [mean, margin_error]


class Metric:
    def __init__(self, fold, true="sentiment", label="results"):
        self.data = fold
        self.results_to_binary()
        self.true = self.data[true]
        self.label = self.data[label]
        self.p = self.precision()
        self.a = self.accuracy()
        self.r = self.recall()

    def results_to_binary(self):
        self.data["results"] = self.data["results"].apply(lambda x: 1 if x > 0.5 else 0)

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


class ConfidenceInterval:
    def __init__(self, array, confidence=0.95):
        self.array = array
        self.confidence = confidence
        self.mean = np.mean(self.array)
        self.std_err = stats.sem(self.array)
        self.margin_error = self.std_err * stats.t.ppf(
            (1 + self.confidence) / 2, len(self.array) - 1
        )

    def get_ci(self):
        return self.mean, self.margin_error


def plot_confidence_interval(
    ax,
    xi,
    mean,
    margin_error,
    color="black",
    hlw=0.1,
):
    left = xi - hlw / 2
    top = mean + margin_error
    right = xi + hlw / 2
    bottom = mean - margin_error
    ax.plot([xi, xi], [top, bottom], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(xi, mean, "x", color="blue")


class ConfidenceGraph:
    def __init__(self, fold_parsers_list):
        self.fold_parsers_list = fold_parsers_list
        self.metrics_names = ["Precision", "Accuracy", "Recall"]
        self.figure = None
        self.axes = None

    def plot_confidence_intervals(self):
        num_subplots = len(self.fold_parsers_list)
        self.figure, self.axes = plt.subplots(
            1, num_subplots, figsize=(12, 6), sharey=True
        )

        for i, (fold_parser, ax) in enumerate(zip(self.fold_parsers_list, self.axes)):
            ticks = [1, 2, 3]
            ax.set_xticks(ticks)
            ax.set_xticklabels(self.metrics_names)

            for j, tic in enumerate(ticks):
                mean = fold_parser.par_values[j][0]
                margin_error = fold_parser.par_values[j][1]
                plot_confidence_interval(ax, j + 1, mean, margin_error)

            ax.set_title(fold_parser.name)

        plt.tight_layout()
        plt.show()


def test_graph():
    # Example usage
    file_1 = "amazon_with_llm_results_with_results.csv"
    file_2 = "amazon_with_results.csv"
    file_3 = "gold_with_results.csv"
    file_4 = "gold_with_llm_results_with_results.csv"

    p_1 = FoldParser(file_1)
    p_2 = FoldParser(file_2)
    p_3 = FoldParser(file_3)
    p_4 = FoldParser(file_4)

    parsed_list = [p_1, p_2, p_3, p_4]

    confidence_graph = ConfidenceGraph(parsed_list)
    confidence_graph.plot_confidence_intervals()


test_graph()
