from scipy import stats
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score

from ensemble.weak import (SVMClassifier, LogisticRegressionClassifier, RandomForestClassifierWrapper,
                           NaiveBayesClassifier)


def perform_anova(y_true, y_preds):
    f_scores, p_values = [], []

    for y_pred in y_preds:
        f_score, p_value = stats.f_oneway(y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4])
        f_scores.append(f_score)
        p_values.append(p_value)

    return f_scores, p_values


class ClassifierComparison:

    def __init__(self, classifiers, dataset):
        self.classifiers = classifiers
        self.dataset = dataset
        self.x = np.vstack(self.dataset['vector'].values)
        self.y = np.vstack(self.dataset['sentiment'].values)
        self.y = self.y.ravel()
        self.results = {}

    def compare_classifiers(self):
        for classifier_name, classifier in self.classifiers.items():
            print(f"Evaluating {classifier_name}")
            y_preds = []

            for _ in range(5):  # Five-fold cross-validation
                y_pred = cross_val_predict(classifier.model, self.x, self.y, cv=5)
                y_preds.append(y_pred)

            self.results[classifier_name] = {
                'f_scores': [],
                'p_values': [],
                'accuracy_scores': []
            }

            self.results[classifier_name]['accuracy_scores'] = cross_val_score(
                classifier.model, self.x, self.y, cv=5, scoring='accuracy')

            f_scores, p_values = perform_anova(self.y, y_preds)
            self.results[classifier_name]['f_scores'] = f_scores
            self.results[classifier_name]['p_values'] = p_values

    def print_results(self):
        for classifier_name, result in self.results.items():
            print(f"Results for {classifier_name}")
            print("Mean Accuracy:", np.mean(result['accuracy_scores']))
            print("Mean F-Score:", np.mean(result['f_scores']))
            print("Mean p-value:", np.mean(result['p_values']))
            print()


# Define your classifiers here
classifiers = {
    'SVM': SVMClassifier(dataset),
    'Naive Bayes': NaiveBayesClassifier(dataset),
    'Logistic Regression': LogisticRegressionClassifier(dataset),
    'Random Forest': RandomForestClassifierWrapper(dataset)
}

# Create the comparison object
comparison = ClassifierComparison(classifiers, dataset)

# Perform comparison and print results
comparison.compare_classifiers()
comparison.print_results()
