from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


class WeakClassifier:

    def __init__(self, dataset, random_state=42):
        self.dataset = dataset
        self.random_state = random_state
        self.x = np.vstack(self.dataset['vector'].values)
        self.y = np.vstack(self.dataset['sentiment'].values)
        self.y = self.y.ravel()
        self.model = None

    def fit_and_evaluate(self):
        # Perform five-fold cross-validation and get predicted labels for each fold
        y_pred = cross_val_predict(self.model, self.x, self.y, cv=5)

        # Calculate accuracy using cross_val_score
        accuracy_scores = cross_val_score(self.model, self.x, self.y, cv=5, scoring='accuracy')
        mean_accuracy = np.mean(accuracy_scores)

        print("Mean Accuracy: %s" % mean_accuracy)

        return y_pred

    def get_results(self):
        return self.fit_and_evaluate()


class SVMClassifier(WeakClassifier):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()

    def load_model(self, kernel='rbf'):
        self.model = SVC(kernel=kernel, random_state=self.random_state)


class NaiveBayesClassifier(WeakClassifier):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()

    def load_model(self):
        self.model = MultinomialNB()


class LogisticRegressionClassifier(WeakClassifier):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()

    def load_model(self):
        self.model = LogisticRegression(random_state=self.random_state)


class RandomForestClassifierWrapper(WeakClassifier):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.load_model()

    def load_model(self):
        self.model = RandomForestClassifier(random_state=self.random_state)



