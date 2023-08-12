from pipeline import Pipe, load_pipe
from llm.util import *
from ensemble.xgboost import *
import time
import pandas as pd


def process_pipe(file="data/gold.csv", save=False, shrink=False):
    my_pipe = Pipe(file, shrink=shrink)
    my_pipe.run_pipe()
    print(my_pipe.processed)
    if save:
        my_pipe.save()
    return my_pipe


def train_test(
    file,
    name,
    include_llm=False,
    include_weak=False,
    shrink=False,
    multi=True,
    depth=3,
    folded=False,
):
    start = time.time()
    # pd.set_option('display.max_colwidth', None)
    cyan("starting to process %s" % file)
    pipe = process_pipe(file, shrink=shrink)
    cyan("initializing classifier for %s" % name)
    xgb_class = Classifier(
        pipe.processed,
        name=name,
        include_llm=include_llm,
        include_weak=include_weak,
        multi=multi,
        max_depth=depth,
        folded=folded,
    )
    # Create a Classifier object
    cyan("starting to test %s" % name)
    pd.set_option("display.max_colwidth", None)
    xgb_class.train()
    end = time.time()
    run_time = end - start
    green("program finished in %f seconds." % run_time)


def load_pickled_pipe(file="data.pickle"):
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


def full():
    train_test(
        "labeled_data/amazon_cells_labelled_labeled.csv", "amazon", include_llm=True
    )
    train_test("labeled_data/yelp_labelled_labeled.csv", "yelp", include_llm=True)
    train_test("labeled_data/imdb_labelled_labeled.csv", "imdb", include_llm=True)
    train_test("labeled_data/gold_labeled.csv", "gold", include_llm=True)
    train_test("labeled_data/amazon_cells_labelled_labeled.csv", "amazon")
    train_test("labeled_data/yelp_labelled_labeled.csv", "yelp")
    train_test("labeled_data/imdb_labelled_labeled.csv", "imdb")
    train_test("labeled_data/gold_labeled.csv", "gold")


def the_weak_full():
    train_test(
        "results\\with_weak\\movies.csv", "movies_weak", include_llm=True, include_weak=True,
        depth=6, multi=False)
    train_test("results\\with_weak\\gold.csv", "gold_weak", include_llm=True, include_weak=True,
               depth=6, multi=False)


def the_movies(shrink=True, multi=False):
    for i in range(1, 11):
        train_test(
            "labeled_data/new_movies_labeled.csv",
            "movies_1000",
            include_llm=True,
            shrink=shrink,
            multi=multi,
            depth=i,
        )
        train_test(
            "labeled_data/new_movies_labeled.csv",
            "movies_1000",
            include_llm=False,
            shrink=shrink,
            multi=multi,
            depth=i,
        )


if __name__ == "__main__":

    the_weak_full()
