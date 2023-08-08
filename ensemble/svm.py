from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


class SVMClassifier(SVC):

    def __init__(self, dataset, kernel='rbf', random_state=42):
        super().__init__(kernel=kernel, random_state=random_state)
        self.dataset = dataset
        self.random_state = random_state
        self.x = np.vstack(self.dataset['vector'].values)
        self.y = np.vstack(self.dataset['sentiment'].values)
        self.y = self.y.ravel()

        # Perform five-fold cross-validation and get predicted labels for each fold
        self.y_pred = cross_val_predict(self, self.x, self.y, cv=5)

        # Calculate accuracy using cross_val_score
        self.accuracy_scores = cross_val_score(self, self.x, self.y, cv=5, scoring='accuracy')
        self.mean_accuracy = np.mean(self.accuracy_scores)

        print("Mean Accuracy: %s" % self.mean_accuracy)

    def get_results(self):
        return self.y_pred
