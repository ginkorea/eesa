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
    """

    def __init__(self, dataset_path: str, use_llm: bool = False, verbose: bool = False):
        self.dataset_path = dataset_path
        self.use_llm = use_llm
        self.verbose = verbose
        self.df = pd.read_csv(dataset_path, sep="|")
        self.X = self.df["sentence"]
        self.y = self.df["sentiment"] if "sentiment" in self.df.columns else None
        self.stop_words = set_up_stop_words()
        self.pipeline = None
        self._build_pipeline()

        if self.verbose:
            cyan(f"[Data] Loaded {len(self.df)} records from {dataset_path}")
            if self.use_llm:
                yellow("[LLM] LLM features enabled.")

    def _build_pipeline(self):
        text_pipeline = Pipeline([
            ("preprocess", TextPreprocessor(self.stop_words)),
            ("vectorize", FeatureVectorizer())
        ])

        if self.use_llm:
            combined = FeatureUnion([
                ("vector", text_pipeline),
                ("llm", LLMLabeler(verbose=self.verbose))
            ])
            features = combined
        else:
            features = text_pipeline

        self.pipeline = Pipeline([
            ("features", features),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        if self.verbose:
            cyan("[Pipeline] Built feature + classifier pipeline")

    def train(self):
        if self.y is None:
            raise ValueError("Target labels (y) not found in dataset.")
        cyan("→ Training model...")
        self.pipeline.fit(self.X, self.y)
        green("✓ Training complete.")

    def predict(self, text_list: list[str]):
        return self.pipeline.predict(text_list)

    def evaluate(self):
        if self.y is None:
            raise ValueError("Cannot evaluate without true labels.")
        preds = self.pipeline.predict(self.X)
        accuracy = np.mean(preds == self.y)
        green(f"✓ Accuracy on training set: {accuracy:.4f}")
        return accuracy

    def label_and_save(self, output_dir="labeled_data/", batch_size=100, start=0):
        os.makedirs(output_dir, exist_ok=True)
        end = min(len(self.X), start + batch_size)
        batch = self.X[start:end]
        cyan(f"→ Labeling batch {start}-{end} with LLM...")

        outputs = [label_row(text, verbose=self.verbose) for text in batch]
        df_out = pd.DataFrame(outputs, columns=[
            "sentiment_score", "confidence_rating", "explanation_score", "explanation"
        ])
        df_out["sentence"] = batch.values
        out_path = f"{output_dir}/batch_{start}_{end}.csv"
        df_out.to_csv(out_path, sep="|", index=False)
        green(f"✓ Saved labeled batch to {out_path}")
        return df_out

    def save_model(self, path="trained_pipeline.pkl"):
        import joblib
        joblib.dump(self.pipeline, path)
        green(f"✓ Pipeline saved to {path}")
