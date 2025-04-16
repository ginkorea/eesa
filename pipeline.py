# pipeline.py

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from ensemble.llm_classifier import label_row
from util import cyan, green, yellow
from preprocessing import (
    TextPreprocessor,
    FeatureVectorizer,
    LLMLabeler,
    MultiExtractor,
    set_up_stop_words
)


class SentimentPipeline:
    """
    A modular sentiment classification pipeline using sklearn's Pipeline API.

    This pipeline can operate in two modes:
    - Traditional NLP vectorization (BoW, TF-IDF, N-Grams)
    - Augmented with LLM-based sentiment embeddings (score, confidence, explanation rating)

    Attributes:
        dataset_path (str): Path to the input dataset CSV
        use_llm (bool): Whether to include LLM-based sentiment scores as features
        stop_words (set): List of stopwords used in preprocessing
        pipeline (Pipeline): Sklearn-compatible pipeline
        df (DataFrame): Loaded dataset
        X (Series): Text input (sentences)
        y (Series): Target labels (sentiment), if present
    """

    def __init__(self, dataset_path: str, use_llm: bool = False):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, sep="|")
        self.X = self.df["sentence"]
        self.y = self.df["sentiment"] if "sentiment" in self.df.columns else None
        self.use_llm = use_llm
        self.stop_words = set_up_stop_words()
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self):
        """
        Internal method to construct the sklearn pipeline.
        Includes either just text-based features or both text and LLM features.
        """
        # Standard NLP pipeline
        text_pipeline = Pipeline([
            ("preprocess", TextPreprocessor(self.stop_words)),
            ("vectorize", FeatureVectorizer())
        ])

        # Optional: combine with LLM feature extractor
        if self.use_llm:
            combined = FeatureUnion([
                ("vector", text_pipeline),
                ("llm", LLMLabeler())
            ])
            features = combined
        else:
            features = text_pipeline

        # Append classifier
        self.pipeline = Pipeline([
            ("features", features),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def train(self):
        """
        Trains the pipeline on the loaded dataset.
        """
        if self.y is None:
            raise ValueError("Target labels (y) not found in dataset.")
        cyan("Training model...")
        self.pipeline.fit(self.X, self.y)
        green("Training complete.")

    def predict(self, text_list: list[str]):
        """
        Predicts sentiment class labels for the provided input texts.

        Args:
            text_list (list[str]): List of raw input sentences

        Returns:
            list[int]: Predicted class labels
        """
        return self.pipeline.predict(text_list)

    def evaluate(self):
        """
        Evaluates the pipeline on the training dataset (if labels available).

        Returns:
            float: Accuracy score on the training set
        """
        if self.y is None:
            raise ValueError("Cannot evaluate without true labels.")
        preds = self.pipeline.predict(self.X)
        accuracy = np.mean(preds == self.y)
        green(f"Accuracy on training set: {accuracy:.4f}")
        return accuracy

    def label_and_save(self, output_dir="labeled_data/", batch_size=100, start=0):
        """
        Runs LLM-based sentiment scoring for a batch of records and saves to CSV.

        Args:
            output_dir (str): Output directory for labeled results
            batch_size (int): Number of records to process
            start (int): Index to start batch from

        Returns:
            DataFrame: DataFrame of labeled outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        end = min(len(self.X), start + batch_size)
        batch = self.X[start:end]
        cyan(f"Labeling batch {start}-{end} with LLM...")

        outputs = [label_row(text) for text in batch]
        df_out = pd.DataFrame(outputs, columns=[
            "sentiment_score", "confidence_rating", "explanation_score", "explanation"
        ])
        df_out["sentence"] = batch.values
        df_out.to_csv(f"{output_dir}/batch_{start}_{end}.csv", sep="|", index=False)
        green(f"Saved labeled batch to {output_dir}/batch_{start}_{end}.csv")
        return df_out

    def save_model(self, path="trained_pipeline.pkl"):
        """
        Saves the trained sklearn pipeline using joblib.

        Args:
            path (str): Path to output file (should end in .pkl)
        """
        import joblib
        joblib.dump(self.pipeline, path)
        green(f"Pipeline saved to {path}")
