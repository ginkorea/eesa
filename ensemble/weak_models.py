# weak_models.py

"""
Modular wrappers for traditional ML models used as weak classifiers in sentiment ensembles.
Supports:
- Cross-validation scoring
- LLM sentiment score augmentation
- Standard sklearn estimator pattern
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from util import cyan, green, yellow
from ensemble.xgb_sentiment import include_llm_vector
from sklearn.metrics import make_scorer, precision_score, recall_score


# Define safe scorers with zero_division fallback
safe_precision = make_scorer(precision_score, zero_division=0)
safe_recall = make_scorer(recall_score, zero_division=0)


class WeakClassifier(BaseEstimator):
    def __init__(
        self,
        dataset,
        model,
        name="WeakClassifier",
        column="WC",
        include_llm=False,
        random_state=42,
    ):
        self.name = name
        self.column = column
        self.include_llm = include_llm
        self.dataset = dataset
        self.random_state = random_state
        self.model = model
        self._prepare()

    def _prepare(self):
        if self.include_llm:
            self.transform_senti_score()
            self.dataset["vector"] = self.dataset.apply(include_llm_vector, axis=1)
        self.X = np.vstack(self.dataset["vector"].values)
        self.y = np.vstack(self.dataset["sentiment"].values).ravel()
        self.cv = self.dataset["fold"].nunique()

    def transform_senti_score(self):
        self.dataset["sentiment_score"] = self.dataset["sentiment_score"].apply(
            lambda x: (x + 1) / 2
        )
        cyan("Transformed sentiment score to 0–1 range.")

    def fit_and_evaluate(self, verbose=False):
        cyan(f"Cross-validating {self.name}...")
        y_pred = cross_val_predict(self.model, self.X, self.y, cv=self.cv)

        acc = cross_val_score(
            self.model, self.X, self.y, cv=self.cv, scoring="accuracy"
        ).mean()
        prec = cross_val_score(
            self.model, self.X, self.y, cv=self.cv, scoring=safe_precision
        ).mean()
        recall = cross_val_score(
            self.model, self.X, self.y, cv=self.cv, scoring=safe_recall
        ).mean()

        yellow(f"{self.name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {recall:.4f}")
        return y_pred, acc, prec, recall, self.column


def SVMClassifier(dataset, kernel="rbf", **kwargs):
    model = SVC(
        kernel=kernel, probability=True, random_state=kwargs.get("random_state", 42)
    )
    return WeakClassifier(
        dataset, model, name="Support Vector Machine", column="SVM", **kwargs
    )


def NaiveBayesClassifier(dataset, **kwargs):
    model = MultinomialNB()
    return WeakClassifier(dataset, model, name="Naive Bayes", column="NB", **kwargs)


def LogisticRegressionClassifier(dataset, **kwargs):
    model = LogisticRegression(random_state=kwargs.get("random_state", 42))
    return WeakClassifier(
        dataset, model, name="Logistic Regression", column="LR", **kwargs
    )


def RandomForestClassifierWrapper(dataset, **kwargs):
    model = RandomForestClassifier(random_state=kwargs.get("random_state", 42))
    return WeakClassifier(dataset, model, name="Random Forest", column="RF", **kwargs)


def apply_weak_classifiers(_df, verbose=False):
    from sklearn.model_selection import KFold

    _df = _df.copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    _df["fold"] = -1
    for i, (_, _test_inx) in enumerate(kf.split(_df)):
        _df.loc[_test_inx, "fold"] = i

    for clf_func in [
        SVMClassifier,
        NaiveBayesClassifier,
        LogisticRegressionClassifier,
        RandomForestClassifierWrapper,
    ]:
        _clf = clf_func(_df)
        predictions, *_ = _clf.fit_and_evaluate(verbose=verbose)
        _df[_clf.column] = predictions
    return _df
