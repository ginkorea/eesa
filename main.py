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


def test():
    start = time.time()
    pd.set_option('display.max_colwidth', None)
    pipe = process_pipe()
    # pipe = load_pickled_pipe(file='gold')
    xgb = Classifier(pipe.processed)
    xgb.train()
    xgb.print_results()
    end = time.time()
    run_time = end - start
    green("program finished in %f seconds." % run_time)


def load_pickled_pipe(file='data.pickle'):
    my_pipe = load_pipe(file)
    print(my_pipe.processed)
    return my_pipe


def label(file='data/gold.csv'):
    start = time.time()
    my_pipe = Pipe(file)
    my_pipe.label_by_batch(batch_size=25, batch_start=550)
    end = time.time()
    run_time = end - start
    green("program finished in %f seconds." % run_time)


if __name__ == "__main__":

    label(file='data/new_movies.csv')
