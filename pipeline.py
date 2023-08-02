from preprocessing.feature_extraction import *
from preprocessing.language_processing import *
from llm.sentiment import *
from data.data import *
from copy import copy
import pickle
import os


class Pipe:

    def __init__(self, dataset):
        self.stop_words = set_up_stop_words()
        if dataset:
            self.name = dataset
            self.raw = load(dataset)
        else:
            self.name = None
            self.raw = None
        self.text = None
        self.processed = None
        self.extractor = None

    def __reduce__(self):
        # Return a tuple with the callable, arguments, and the state (all attributes needed for recreation)
        return (self.__class__, (self.name,), {
            'text': self.text,
            'processed': self.processed,
            'extractor': self.extractor,
            'stop_words': self.stop_words
        })

    def process_texts(self):
        self.processed = copy(self.raw)
        self.processed = self.processed.assign(processed=self.processed['sentence'])
        self.processed['processed'] = self.processed['processed'].apply(process_text, args=(self.stop_words,))
        self.text = list(self.processed['processed'])
        for i, t in enumerate(self.text):
            self.text[i] = ' '.join(t)
        # print("self.text: length: %s type: %s" % (len(self.text), type(self.text)))


    def extract_features(self):
        self.extractor = MultiExtractor(self.text)
        self.extractor.fit()
        self.extractor.process()
        self.processed['vector'] = self.extractor.vector_list

    def run_pipe(self):
        self.process_texts()
        self.extract_features()

    def save(self):
        # Extract the filename after 'data/' and remove the '.csv' extension
        file_name = os.path.basename(self.name)
        file_name = file_name.replace('.csv', '')
        save_dir = "pickle/"  # Specify the directory path where you want to save the data
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Save raw data as CSV
        cyan("starting to save...")
        raw_csv_path = os.path.join(save_dir, f"{file_name}_raw.csv")
        self.raw.to_csv(raw_csv_path, index=False)

        # Save text data as CSV
        text_csv_path = os.path.join(save_dir, f"{file_name}_text.csv")
        pd.DataFrame({"text": self.text}).to_csv(text_csv_path, index=False)

        # Save processed data as CSV
        processed_csv_path = os.path.join(save_dir, f"{file_name}_processed.csv")
        self.processed.to_csv(processed_csv_path, index=False)

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
    pipe.extractor.vector_list = pipe.processed['vector']

    return pipe
