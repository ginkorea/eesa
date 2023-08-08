import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from llm.util import *
import concurrent.futures
import os


class Classifier:

    def __init__(self, processed_data, n_splits=5, shuffle=True, random_state=23, max_depth=3,
                 objective='binary:logistic', eval_metric='logloss', name='test', include_llm=False, multi=False):
        self.processed_data = processed_data
        self.processed_data['results'] = np.zeros(len(self.processed_data))
        self.processed_data['fold'] = np.zeros((len(self.processed_data)))
        if include_llm:
            self.processed_data['vector'] = self.processed_data.apply(include_llm_vector, axis=1)
            name = name + "_with_llm_results"
        # yellow(self.processed_data)
        self.x = np.vstack(self.processed_data['vector'].values)
        # cyan(self.x)
        self.y = np.vstack(self.processed_data['sentiment'].values)
        # yellow(self.y)
        self.parameters = {
            'objective': objective,
            'eval_metric': eval_metric,
            'max_depth': max_depth,
            'eta': 0.1,
            'seed': random_state
        }
        self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.cv_results = {}  # Store results as dictionary
        self.short_name = name
        self.name = "depth_" + str(max_depth) + "_" + name
        self.multi = multi
        cyan("initialized xgboost classifier for %s" % self.name)

    def _train_fold(self, fold_index, train_index, test_index, boost_rounds):
        cyan("training fold %s" % fold_index)
        x_train, x_test = self.x[train_index], self.x[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]

        d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, label=y_test)

        model = xgb.train(self.parameters, d_train, num_boost_round=boost_rounds)
        y_pred = model.predict(d_test)
        cyan("finished training fold %s" % fold_index)

        return fold_index, y_pred, test_index

    def train(self, boost_rounds=100):
        results = []

        if self.multi:

            with concurrent.futures.ProcessPoolExecutor() as executor:
                fold_results = []
                for fold_index, (train_index, test_index) in enumerate(self.k_fold.split(self.x)):
                    fold_results.append(
                        executor.submit(self._train_fold, fold_index, train_index, test_index, boost_rounds))

                for future in concurrent.futures.as_completed(fold_results):
                    fold_index, y_pred, test_index = future.result()
                    cyan("fold index = %s; y_pred =  %s; test_index= %s" % (fold_index, y_pred, test_index))
                    results.append([fold_index, y_pred, test_index])

        else:

            for fold_index, (train_index, test_index) in enumerate(self.k_fold.split(self.x)):
                fold_index, y_pred, test_index = self._train_fold(fold_index, train_index, test_index, boost_rounds)
                result = [fold_index, y_pred, test_index]
                results.append(result)
                cyan("fold index = %s; y_pred =  %s; test_index= %s" % (fold_index, y_pred, test_index))

        self.cv_results = results
        self.map_results_to_dataframe()  # Map the results to the original DataFrame
        self.save_dataframe()  # Save the entire DataFrame with results and mapping
        green("finished training xbg classifier for %s" % self.name)

    def map_results_to_dataframe(self):
        if self.multi:
            cyan('cv_result: %s' % self.cv_results)
            for result in self.cv_results:
                yellow('result: %s' % result)
                fold_index = result[0]
                y_pred = result[1]
                test_index = result[2]
                self.processed_data.loc[test_index, 'results'] = y_pred
                self.processed_data.loc[test_index, 'fold'] = fold_index
        else:
            for result in self.cv_results:
                fold_index = result[0]
                test_indexes = result[2]
                for i, test_index in enumerate(test_indexes):
                    y_pred = result[1][i]
                    self.processed_data.loc[test_index, 'results'] = y_pred
                    self.processed_data.loc[test_index, 'fold'] = fold_index

    def save_dataframe(self):
        if os == 'nt':
            file = 'results\\' + self.short_name + "\\" + self.name + "_with_results.csv"
        else:
            file = 'results/' + self.short_name + "/" + self.name + "_with_results.csv"

        # Get the directory path from the file path
        dir_path = os.path.dirname(file)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        self.processed_data.to_csv(file, index=False, sep="|")

    def print_results(self):
        cyan(self.cv_results)


def include_llm_vector(row):
    original_vector = row['vector']
    senti_score = row['sentiment_score']
    conf_rate = row['confidence_rating']
    expl_score = row['explanation_score']
    cyan('senti_score: %s; conf_rate: %s; expl_score: %s' % (senti_score, conf_rate, expl_score))
    modified_vector = np.concatenate([original_vector, [senti_score, conf_rate, expl_score]])
    yellow('modified vector: %s' % modified_vector[-5:])
    return modified_vector
