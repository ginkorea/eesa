from preprocessing.feature_extraction import *
from preprocessing.language_processing import *
from llm.sentiment import *
from data.data import *
from copy import copy
import os
from ensemble.gpt_class import label_row
import time
import ensemble.weak as weak


class Pipe:
    def __init__(self, dataset, shrink=False, size=1000, cut=False, cut_size=1000):
        self.stop_words = set_up_stop_words()
        self.name = dataset
        self.raw = load(dataset)
        if cut:
            self.cut(size=cut_size)
        if shrink:
            self.shrink(size=size)
        self.text = None
        self.processed = None
        self.extractor = None
        self.labeled = None

    def __reduce__(self):
        # Return a tuple with the callable, arguments, and the state (all attributes needed for recreation)
        return (
            self.__class__,
            (self.name,),
            {
                "text": self.text,
                "processed": self.processed,
                "extractor": self.extractor,
                "stop_words": self.stop_words,
            },
        )

    def shrink(self, size):
        self.raw = self.raw.sample(n=size, random_state=1)
        cyan("shrank dataset to %s" % len(self.raw))

    def cut(self, size):
        self.raw = self.raw[:size]
        cyan("cut dataset to %s" % len(self.raw))

    def process_texts(self):
        self.processed = copy(self.raw)
        self.processed = self.processed.assign(processed=self.processed["sentence"])
        self.processed["processed"] = self.processed["processed"].apply(
            process_text, args=(self.stop_words,)
        )
        self.text = list(self.processed["processed"])
        for i, t in enumerate(self.text):
            self.text[i] = " ".join(t)
        # print("self.text: length: %s type: %s" % (len(self.text), type(self.text)))

    def label_texts(self, test=False, batch=True, start=0, stop=10):
        cyan("Starting to label text. This will take a while.")
        self.labeled = copy(self.raw)
        if test:
            self.labeled = self.labeled.head(3)
        if batch:
            self.labeled = self.labeled[start:stop]
        self.labeled["output"] = self.labeled.apply(
            lambda row: label_row(row["sentence"]), axis=1
        )
        self.labeled[
            ["sentiment_score", "confidence_rating", "explanation_score", "explanation"]
        ] = pd.DataFrame(self.labeled["output"].tolist(), index=self.labeled.index)
        self.labeled.drop(columns=["output"], inplace=True)
        # cyan(self.labeled)
        self.labeled.to_csv("test_label.csv", index=False, sep="|")
        file_name = os.path.basename(self.name)
        file_name = file_name.replace(".csv", "")
        save_dir = f"labeled_data/{file_name}"  # Specify the directory path where you want to save the data
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        if batch:
            labeled_csv_path = os.path.join(
                save_dir, f"batch_{start}_{stop}_labeled.csv"
            )
        else:
            labeled_csv_path = os.path.join(save_dir, f"full_labeled.csv")
        self.labeled.to_csv(labeled_csv_path, index=False, sep="|")
        return start, stop

    def label_by_batch(self, batch_size=10, batch_start=0):
        final = len(self.raw)
        cyan("starting to process %s records in by batches of %s" % (final, batch_size))
        batch_end = batch_start + batch_size
        while batch_end < final:
            start_time = time.time()
            batch_start, batch_end = self.label_texts(
                batch=True, start=batch_start, stop=batch_end
            )
            end_time = time.time()
            run_time = end_time - start_time
            green(
                "finished processing batch %s - %s in %s seconds"
                % (batch_start, batch_end, run_time)
            )
            batch_start = batch_end
            batch_end += batch_size
        if batch_start < final:
            start_time = time.time()
            batch_start, batch_end = self.label_texts(
                batch=True, start=batch_start, stop=final
            )
            end_time = time.time()
            run_time = end_time - start_time
            green(
                "finished processing final batch %s - %s in %s seconds"
                % (batch_start, final, run_time)
            )

    def extract_features(self):
        self.extractor = MultiExtractor(self.text)
        self.extractor.fit()
        self.extractor.process()
        self.processed["vector"] = self.extractor.vector_list

    def run_pipe(self):
        self.process_texts()
        self.extract_features()

    def create_weak_classifiers(self):
        svm_classifier = weak.SVMClassifier(self.processed)
        naive_bayes_classifier = weak.NaiveBayesClassifier(self.processed)
        logistic_regression_classifier = weak.LogisticRegressionClassifier(
            self.processed
        )
        random_forest_classifier = weak.RandomForestClassifierWrapper(self.processed)

        return (
            svm_classifier,
            naive_bayes_classifier,
            logistic_regression_classifier,
            random_forest_classifier,
        )

    def save(self):
        # Extract the filename after 'data/' and remove the '.csv' extension
        file_name = os.path.basename(self.name)
        file_name = file_name.replace(".csv", "")
        save_dir = "labeled_data/"  # Specify the directory path where you want to save the data
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Save raw data as CSV
        cyan("starting to save...")
        raw_csv_path = os.path.join(save_dir, f"{file_name}_raw.csv")
        self.raw.to_csv(raw_csv_path, index=False, sep="|")

        # Save text data as CSV
        text_csv_path = os.path.join(save_dir, f"{file_name}_text.csv")
        pd.DataFrame({"text": self.text}).to_csv(text_csv_path, index=False, sep="|")

        # Save processed data as CSV
        processed_csv_path = os.path.join(save_dir, f"{file_name}_processed.csv")
        self.processed.to_csv(processed_csv_path, index=False, sep="|")

        print("Data successfully saved.")


def load_pipe(file_name):
    save_dir = "pickle/"  # Specify the directory path where you saved the data
    raw_csv_path = os.path.join(save_dir, f"{file_name}_raw.csv")
    text_csv_path = os.path.join(save_dir, f"{file_name}_text.csv")
    processed_csv_path = os.path.join(save_dir, f"{file_name}_processed.csv")

    raw_data = pd.read_csv(raw_csv_path)
    text_data = pd.read_csv(text_csv_path)["text"].tolist()
    processed_data = pd.read_csv(processed_csv_path)

    pipe = Pipe(None)  # Create an empty Pipe object
    pipe.name = file_name
    pipe.raw = raw_data
    pipe.text = text_data
    pipe.processed = processed_data

    # Refit the extractor
    pipe.extractor = MultiExtractor(pipe.text)
    pipe.extractor.fit()
    pipe.extractor.vector_list = pipe.processed["vector"]

    return pipe


def classify_with_weak(file_name, name):
    pipe = Pipe(file_name, cut=True, cut_size=1000)
    pipe.process_texts()
    pipe.extract_features()
    # pipe2 = copy(pipe)
    cyan("starting without llm")
    svm1, naive1, log1, rf1 = pipe.create_weak_classifiers()
    classifiers1 = [svm1, naive1, log1, rf1]
    weak_results = []
    weak_columns = []
    for classifier in classifiers1:
        yellow("calculating %s without llm" % classifier.column)
        y_pred, accuracy, precision, recall, column = classifier.fit_and_evaluate()
        weak_results.append(y_pred)
        weak_columns.append(column)
    pipe.processed = add_weak_results(pipe.processed, weak_results, weak_columns)
    pipe.processed.to_csv(f"results/with_weak/{name}.csv", index=False, sep="|")
    cyan("starting with llm")
    """
    weak_columns = []
    weak_results = []
    svm2, naive2, log2, rf2 = pipe2.create_weak_classifiers()
    classifiers2 = [svm2, naive2, log2, rf2]
    for classifier in classifiers2:
        cyan("calculating %s with llm" % classifier.column)
        y_pred, accuracy, precision, recall, column = classifier.fit_and_evaluate(
            include_llm=True
        )
        y_pred = np.array(y_pred)
        weak_results.append(y_pred)
        weak_columns.append(column)
    pipe2.processed = add_weak_results(pipe2.processed, weak_results, weak_columns)
    pipe2.processed.to_csv(
        f"results/with_weak/imdb_results_with_llm.csv", index=False, sep="|"
    )
    """


def add_weak_results(df, results, columns):
    for result, column_name in zip(results, columns):
        df[column_name] = result
    return df


def classify_all_with_weak():
    # classify_with_weak("results\\depth_6_imdb_with_results.csv", "imdb")
    # classify_with_weak("results\\depth_6_yelp_with_results.csv", "yelp")
    # classify_with_weak("results\\depth_6_amazon_with_results.csv", "amazon")
    # classify_with_weak("results\\depth_6_gold_with_results.csv", "gold")
    classify_with_weak("results\\depth_7_movies_1000_with_results.csv", "movies")


