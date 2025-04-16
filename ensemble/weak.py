from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from ensemble.xgboost import include_llm_vector
from util import cyan


class WeakClassifier:
    def __init__(self, dataset, name=None, random_state=42):
        if name is None:
            name = "weak_classifier"
        self.name = name
        self.dataset = dataset
        self.random_state = random_state
        self.x = np.vstack(self.dataset["vector"].values)
        self.y = np.vstack(self.dataset["sentiment"].values)
        self.y = self.y.ravel()
        self.model = None
        self.column = None

    def fit_and_evaluate(self, include_llm=False, skip_metrics=True):
        if include_llm:
            self.transform_senti_score()
            self.dataset["vector"] = self.dataset.apply(include_llm_vector, axis=1)
            self.x = np.vstack(self.dataset["vector"].values)
        # Perform cross-validation and get predicted labels for each fold
        cyan("Performing cross-validation for %s" % self.name)
        y_pred = cross_val_predict(
            self.model, self.x, self.y, cv=self.dataset["fold"].nunique()
        )
        cyan("Finished cross-validation for %s" % self.name)
        if not skip_metrics:
            # Calculate accuracy using cross_val_score
            accuracy_scores = cross_val_score(
                self.model,
                self.x,
                self.y,
                cv=self.dataset["fold"].nunique(),
                scoring="accuracy",
            )
            mean_accuracy = np.mean(accuracy_scores)

            # Calculate precision using cross_val_score
            precision_scores = cross_val_score(
                self.model,
                self.x,
                self.y,
                cv=self.dataset["fold"].nunique(),
                scoring="precision",
            )
            mean_precision = np.mean(precision_scores)

            # Calculate recall using cross_val_score
            recall_scores = cross_val_score(
                self.model,
                self.x,
                self.y,
                cv=self.dataset["fold"].nunique(),
                scoring="recall",
            )
            mean_recall = np.mean(recall_scores)

            print("Mean Accuracy: %s" % mean_accuracy)
            print("Mean Precision: %s" % mean_precision)
            print("Mean Recall: %s" % mean_recall)
        else:
            mean_accuracy = None
            mean_precision = None
            mean_recall = None

        return y_pred, mean_accuracy, mean_precision, mean_recall, self.column

    def transform_senti_score(self):
        """Transform the sentiment score to be between 0 and 1"""
        self.dataset["sentiment_score"] = self.dataset["sentiment_score"].apply(
            lambda x: (x + 1) / 2
        )
        cyan("Transformed sentiment score to be between 0 and 1")


class SVMClassifier(WeakClassifier):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()
        self.column = "SVM"
        self.name = "Support Vector Machine"

    def load_model(self, kernel="rbf"):
        self.model = SVC(kernel=kernel, random_state=self.random_state)


class NaiveBayesClassifier(WeakClassifier):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()
        self.column = "NB"
        self.name = "Naive Bayes"

    def load_model(self):
        self.model = MultinomialNB()


class LogisticRegressionClassifier(WeakClassifier):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()
        self.column = "LR"
        self.name = "Logistic Regression"

    def load_model(self):
        self.model = LogisticRegression(random_state=self.random_state)


class RandomForestClassifierWrapper(WeakClassifier):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()
        self.column = "RF"
        self.name = "Random Forest"

    def load_model(self):
        self.model = RandomForestClassifier(random_state=self.random_state)
