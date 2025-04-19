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


class WeakClassifier(BaseEstimator):
    """
    Base wrapper for weak learners with LLM and metric integration.

    Args:
        dataset (pd.DataFrame): Input dataset with vector + sentiment + fold info
        model (sklearn classifier): Any sklearn-compatible model instance
        name (str): Human-readable name
        column (str): Column label to save predictions to
        include_llm (bool): Whether to append LLM features to vector
    """

    def __init__(
        self,
        dataset,
        model,
        name: str = "WeakClassifier",
        column: str = "WC",
        include_llm: bool = False,
        random_state: int = 42
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
        """Convert sentiment score range from [-1,1] to [0,1]."""
        self.dataset["sentiment_score"] = self.dataset["sentiment_score"].apply(lambda x: (x + 1) / 2)
        cyan("Transformed sentiment score to 0–1 range.")

    def fit_and_evaluate(self, verbose=False):
        """
        Run cross-validation predictions and compute basic metrics.

        Returns:
            Tuple: (y_pred, acc, prec, recall, column_name)
        """
        cyan(f"Cross-validating {self.name}...")
        y_pred = cross_val_predict(self.model, self.X, self.y, cv=self.cv)

        if verbose:
            green(f"{self.name} predictions: {y_pred[:5]}...")

        # Compute metrics
        acc = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring="accuracy").mean()
        prec = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring="precision").mean()
        recall = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring="recall").mean()

        yellow(f"{self.name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {recall:.4f}")
        return y_pred, acc, prec, recall, self.column


# === SPECIFIC CLASSIFIER WRAPPERS ===

def SVMClassifier(dataset, kernel="rbf", **kwargs):
    """Returns an SVM-based weak classifier."""
    model = SVC(kernel=kernel, probability=True, random_state=kwargs.get("random_state", 42))
    return WeakClassifier(dataset, model, name="Support Vector Machine", column="SVM", **kwargs)


def NaiveBayesClassifier(dataset, **kwargs):
    """Returns a Naive Bayes weak classifier."""
    model = MultinomialNB()
    return WeakClassifier(dataset, model, name="Naive Bayes", column="NB", **kwargs)


def LogisticRegressionClassifier(dataset, **kwargs):
    """Returns a Logistic Regression weak classifier."""
    model = LogisticRegression(random_state=kwargs.get("random_state", 42))
    return WeakClassifier(dataset, model, name="Logistic Regression", column="LR", **kwargs)


def RandomForestClassifierWrapper(dataset, **kwargs):
    """Returns a Random Forest weak classifier."""
    model = RandomForestClassifier(random_state=kwargs.get("random_state", 42))
    return WeakClassifier(dataset, model, name="Random Forest", column="RF", **kwargs)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import KFold

    print("\n=== Running weak_models test suite ===")

    # Generate synthetic classification data
    x, y = make_classification(n_samples=50, n_features=10, n_classes=2, random_state=42)
    x = np.abs(x)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x))

    # Construct DataFrame
    df = pd.DataFrame({
        "vector": list(x),
        "sentiment": list(y),
        "sentiment_score": [2 * int(label) - 1 for label in y],  # Map [0,1] to [-1,1]
        "fold": 0
    })
    for fold_num, (train_idx, test_idx) in enumerate(folds):
        df.loc[test_idx, "fold"] = fold_num

    classifiers = [
        SVMClassifier(df.copy()),
        NaiveBayesClassifier(df.copy()),
        LogisticRegressionClassifier(df.copy()),
        RandomForestClassifierWrapper(df.copy())
    ]

    for clf in classifiers:
        print(f"\nTesting {clf.name}...")
        y_pred, acc, prec, recall, _ = clf.fit_and_evaluate(verbose=True)
        assert len(y_pred) == len(df), f"{clf.name}: prediction length mismatch"
        assert 0 <= acc <= 1, "Invalid accuracy"
        assert 0 <= prec <= 1, "Invalid precision"
        assert 0 <= recall <= 1, "Invalid recall"

    # Optional: test with LLM vector enabled (if include_llm_vector is defined)
    print("\nTesting with LLM feature integration...")
    clf_llm = LogisticRegressionClassifier(df.copy(), include_llm=True)
    y_pred, *_ = clf_llm.fit_and_evaluate()
    assert len(y_pred) == len(df), "LLM-integrated model failed."

    print("\n✓ All weak classifier tests passed.\n")
