import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from util import *
from tabulate import tabulate
import os


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
        # self.get_confidence_intervals(confidence=confidence)

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
    def __init__(self, fold, true="sentiment", label="results", transform=False):
        self.data = fold
        if transform:
            self.transform_senti_score()
            self.results_to_binary(results="sentiment_score")
        else:
            self.results_to_binary()
        self.true = self.data[true]
        self.label = self.data[label]
        self.p = self.precision()
        self.a = self.accuracy()
        self.r = self.recall()

    def results_to_binary(self, results="results"):
        self.data[results] = self.data[results].apply(lambda x: 1 if x > 0.5 else 0)

    def transform_senti_score(self):
        """Transform the sentiment score to be between 0 and 1"""
        self.data["sentiment_score"] = self.data["sentiment_score"].apply(
            lambda x: (x + 1) / 2
        )

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
        self.lower_bound = self.mean - self.margin_error
        self.upper_bound = self.mean + self.margin_error

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


def senti_score_metrics(file):
    df = pd.read_csv(file, sep="|")
    metrics = Metric(df, true="sentiment", label="sentiment_score", transform=True)
    metrics.results()


def senti_score_test():
    senti_score_metrics("amazon_with_llm_results_with_results.csv")
    senti_score_metrics("yelp_with_llm_results_with_results.csv")
    senti_score_metrics("imdb_with_llm_results_with_results.csv")
    senti_score_metrics("gold_with_llm_results_with_results.csv")
    senti_score_metrics("movies_1000_with_llm_results/depth_3_movies_1000_with_llm_results_with_results.csv")


def results_to_binary(df, results="results"):
    df[results] = df[results].apply(lambda x: 1 if x > 0.5 else 0)
    return df


class ResultsAnalyzer:

    def __init__(self, dataset, name=None, true="sentiment", label="results", confidence=0.95):
        self.dataset = dataset
        if name is None:
            self.name = dataset
            self.name = self.name.split("/")[-1]
            self.name = self.name.split(".")[0]
        self.true = true
        self.label = label
        red(self.label)
        self.sentiment_score = "sentiment_score"
        self.confidence = confidence
        self.fp = FoldParser(self.dataset, numb_folds=5, true=self.true, label=self.label, confidence=self.confidence)

    def get_examples(self, true, pred, number, sentiment_score=False):
        df = self.fp.data
        cyan(df)
        df = df[df[self.true] == true].copy()
        cyan(df)
        if not sentiment_score:
            df = results_to_binary(df, results=self.label)
            df = df[df[self.label] == pred].copy()
        else:
            df = results_to_binary(df, results=self.sentiment_score)
            df = df[df[self.sentiment_score] == pred].copy()
        df.round(2)
        df.drop(columns=['vector', 'processed', 'fold', 'confidence_rating', 'explanation_score', 'results'], inplace=True)
        sampled_rows = min(number, len(df))
        df = df.sample(sampled_rows)
        cyan(tabulate(df, headers='keys', tablefmt='psql'))
        column_format = "r{1cm} p{0.4in} r{1cm} p{0.4in}"
        df.to_latex(f"LaTeX\\{self.name}_{true}_{pred}.tex", index=False, column_format=column_format)
        return df

    def plot_correlation_matrix(self):
        df = self.fp.data
        df.drop(columns=['vector', 'processed', 'fold', 'sentence','explanation'], inplace=True)
        correlation_matrix = df.corr()

        # Create a plot with annotated values
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Correlation Coefficient')
        plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
        plt.title('Correlation Matrix with Annotated Values')
        plt.show()

    def plot_scatter(self):
        df = self.fp.data
        df.drop(columns=['vector', 'processed', 'fold', 'sentence','explanation'], inplace=True)
        df = results_to_binary(df, results=self.label)
        df['accuracy'] = (df[self.true] == df[self.label])

        # Create scatter plot with color and size encoding
        plt.figure(figsize=(10, 6))

        # Scatter plot with color and size encoding for accuracy
        scatter = plt.scatter(
            df['confidence_rating'],
            df['explanation_score'],
            c=df['accuracy'],
            cmap='coolwarm',
            s=df['accuracy'] * 100 + 50,  # Adjust size based on accuracy
            alpha=0.7,  # Set transparency for better visibility
        )

        plt.colorbar(scatter, label='Accuracy')
        plt.xlabel('Confidence Rating')
        plt.ylabel('Explanation Score')
        plt.title('Scatter Plot: Confidence Rating vs. Explanation Score with Accuracy Color and Size Encoding')

        plt.show()

    def plot_covariance_matrix(self):
        df = self.fp.data
        df.drop(columns=['vector', 'processed', 'fold', 'sentence', 'explanation'], inplace=True)
        covariance_matrix = df.cov()

        # Create a plot with annotated values
        plt.figure(figsize=(8, 6))
        plt.imshow(covariance_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Covariance Value')
        plt.xticks(range(len(covariance_matrix)), covariance_matrix.columns, rotation=45)
        plt.yticks(range(len(covariance_matrix)), covariance_matrix.columns)
        plt.title('Covariance Matrix with Annotated Values')
        plt.show()



def merge_csv_files(directory, output_filename="merged.csv"):
    """
    Merge all CSV files in a directory into one large CSV file.

    Parameters:
    - directory: Path to the directory containing the CSV files.
    - output_filename: Name of the output merged CSV file.

    Returns:
    - Full path to the merged CSV file.
    """
    # List all files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    # Use pandas to concatenate all CSVs into one dataframe
    combined_csv = pd.concat([pd.read_csv(f, sep="|") for f in all_files])

    # Write the combined dataframe to a new CSV
    output_path = os.path.join(directory, output_filename)
    combined_csv.to_csv(output_path, index=False, sep="|")

    return output_path


def analyze_directory_results(directory, label="sentiment_score"):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)

            # Create an instance of ResultsAnalyzer
            analyzer = ResultsAnalyzer(filepath, label=label)

            # Getting 10 samples for each category
            analyzer.get_examples(true=1, pred=1, number=10)  # True Positives
            analyzer.get_examples(true=0, pred=0, number=10)  # True Negatives
            analyzer.get_examples(true=1, pred=0, number=10)  # False Negatives
            analyzer.get_examples(true=0, pred=1, number=10)  # False Positives


analyze_directory_results("depth_3_llm\\")
