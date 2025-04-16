# anova_analysis.py

"""
Performs statistical comparison of multiple weak classifiers using ANOVA and accuracy metrics.
"""

import numpy as np
from scipy import stats
from typing import Dict
from sklearn.model_selection import cross_val_predict, cross_val_score
from ensemble.weak_models import (
    SVMClassifier,
    NaiveBayesClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifierWrapper
)
from util import cyan, green


def perform_anova(predictions: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """
    Performs one-way ANOVA on multiple classifier prediction sets.

    Args:
        predictions (list): List of classifier prediction arrays

    Returns:
        tuple: (list of F-scores, list of p-values)
    """
    f_scores, p_values = [], []
    for pred_set in predictions:
        f_score, p_value = stats.f_oneway(*pred_set)
        f_scores.append(f_score)
        p_values.append(p_value)
    return f_scores, p_values


class ClassifierComparison:
    """
    Compares multiple classifiers using cross-validation and ANOVA.

    Args:
        classifiers (Dict[str, WeakClassifier]): Dictionary of classifier instances
        dataset (pd.DataFrame): Dataset with 'vector' and 'sentiment' columns
        folds (int): Number of folds for cross-validation
    """

    def __init__(self, classifiers: Dict[str, object], dataset, folds: int = 5):
        self.classifiers = classifiers
        self.dataset = dataset
        self.x = np.vstack(dataset["vector"].values)
        self.y = np.vstack(dataset["sentiment"].values).ravel()
        self.k = folds
        self.results = {}

    def compare(self):
        """
        Runs evaluation across all classifiers and stores results.
        """
        for name, clf in self.classifiers.items():
            cyan(f"Evaluating {name}")
            predictions = []
            for _ in range(self.k):
                y_pred = cross_val_predict(clf.model, self.x, self.y, cv=self.k)
                predictions.append(y_pred)

            accuracy_scores = cross_val_score(clf.model, self.x, self.y, cv=self.k, scoring="accuracy")
            f_scores, p_values = perform_anova(self.y, [predictions])

            self.results[name] = {
                "accuracy_scores": accuracy_scores,
                "f_scores": f_scores,
                "p_values": p_values
            }

    def print_summary(self):
        """
        Prints a performance summary for each classifier.
        """
        for name, result in self.results.items():
            green(f"Results for {name}")
            print(f"  Mean Accuracy  : {np.mean(result['accuracy_scores']):.4f}")
            print(f"  Mean F-Score   : {np.mean(result['f_scores']):.4f}")
            print(f"  Mean p-value   : {np.mean(result['p_values']):.6f}")
            print("-" * 40)


def run_comparison_on_dataset(dataset):
    """
    Builds classifiers and runs ANOVA comparison on the given dataset.

    Args:
        dataset (pd.DataFrame): Input with 'vector' and 'sentiment' columns
    """
    classifiers = {
        "SVM": SVMClassifier(dataset),
        "Naive Bayes": NaiveBayesClassifier(dataset),
        "Logistic Regression": LogisticRegressionClassifier(dataset),
        "Random Forest": RandomForestClassifierWrapper(dataset)
    }

    comparison = ClassifierComparison(classifiers, dataset)
    comparison.compare()
    comparison.print_summary()
