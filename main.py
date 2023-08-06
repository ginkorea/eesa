from pipeline import Pipe, load_pipe
from llm.util import *
from ensemble.xgboost import *
import time
import pandas as pd


def process_pipe(file='data/gold.csv', save=False):
    my_pipe = Pipe(file)
    my_pipe.run_pipe()
    print(my_pipe.processed)
    if save:
        my_pipe.save()
    return my_pipe


def train_test(file, name, include_llm=False):
    start = time.time()
    pd.set_option('display.max_colwidth', None)
    pipe = process_pipe(file)
    xgb_class = Classifier(pipe.processed, name=name, include_llm=include_llm)
    xgb_class.train()
    end = time.time()
    run_time = end - start
    green("program finished in %f seconds." % run_time)


def load_pickled_pipe(file='data.pickle'):
    my_pipe = load_pipe(file)
    print(my_pipe.processed)
    return my_pipe


def label(file, batch_size, batch_start):
    start = time.time()
    green("creating pipe for %s, size: %s, start: %s" % (file, batch_size, batch_start))
    my_pipe = Pipe(file)
    my_pipe.label_by_batch(batch_size=batch_size, batch_start=batch_start)
    end = time.time()
    run_time = end - start
    green("program finished in %f seconds." % run_time)


if __name__ == "__main__":

    train_test('labeled_data/amazon_cells_labelled_labeled.csv', 'amazon', include_llm=True)
    train_test('labeled_data/yelp_labelled_labeled.csv', 'yelp', include_llm=True)
    train_test('labeled_data/imdb_labelled_labeled.csv', 'imdb', include_llm=True)
    train_test('labeled_data/gold_labeled.csv', 'gold', include_llm=True)
    train_test('labeled_data/amazon_cells_labelled_labeled.csv', 'amazon')
    train_test('labeled_data/yelp_labelled_labeled.csv', 'yelp')
    train_test('labeled_data/imdb_labelled_labeled.csv', 'imdb')
    train_test('labeled_data/gold_labeled.csv', 'gold')
