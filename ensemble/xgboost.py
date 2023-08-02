import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from llm.util import *
import multiprocessing


class Classifier:

    def __init__(self, processed_data, n_splits=5, shuffle=True, random_state=23, max_depth=3,
                 objective='binary:logistic', eval_metric='logloss', from_save=False):
        if from_save:
            self.x = np.vstack([np.fromstring(arr_str[1:-1], sep=' ') for arr_str in processed_data['vector']])
        else:
            self.x = np.vstack(processed_data['vector'].values)
        cyan(self.x)
        self.y = np.vstack(processed_data['sentiment'].values)
        yellow(self.y)
        self.parameters = {
            'objective': objective,  # for binary classification
            'eval_metric': eval_metric,  # logloss for binary classification
            'max_depth': max_depth,  # maximum depth of the tree
            'eta': 0.1,  # learning rate
            'seed': random_state  # random seed for reproducibility
        }
        # Perform k-fold cross-validation
        self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.cv_results = []  # To store the cross-validation results for each fold

    def _train_fold_with_dict(self, fold_index, train_index, test_index, boost_rounds, result_dict):
        x_train, x_test = self.x[train_index], self.x[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]

        # Convert the NumPy arrays to DMatrix format
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, label=y_test)

        # Train the XGBoost model
        model = xgb.train(self.parameters, d_train, num_boost_round=boost_rounds)

        # Make predictions on the test set
        y_pred = model.predict(d_test)

        # Store the results in the shared dictionary
        result_dict[fold_index] = y_pred

    def train(self, boost_rounds=100):
        i = 0
        manager = multiprocessing.Manager()
        result_dict = manager.dict()
        processes = []
        for fold_index, (train_index, test_index) in enumerate(self.k_fold.split(self.x)):
            process = multiprocessing.Process(target=self._train_fold_with_dict,
                                              args=(fold_index, train_index, test_index, boost_rounds, result_dict))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Sort and collect results from the shared dictionary
        results = [None] * len(processes)
        for fold_index in result_dict:
            results[fold_index] = result_dict[fold_index]

        self.cv_results = results
        green("finished training model")

    def print_results(self):
        cyan(self.cv_results)
