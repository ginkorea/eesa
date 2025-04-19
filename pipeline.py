# pipeline.py
# Custom pipeline for sentiment analysis using XGBoost and optional LLM/weak classifiers.

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from util import cyan, green, yellow, red
from preprocessing import TextPreprocessor, FeatureVectorizer, set_up_stop_words
from ensemble.llm_classifier import label_row
from ensemble.weak_models import apply_weak_classifiers
from pipeline_utils import (
    extract_llm_features,
    extract_weak_features,
    noop_return_empty,
)


class SentimentPipeline:
    def __init__(self, dataset_path: str, use_llm=False, use_weak=False, verbose=False):
        self.dataset_path = dataset_path
        self.use_llm = use_llm
        self.use_weak = use_weak
        self.verbose = verbose
        self.stop_words = set_up_stop_words()

        self.df = pd.read_csv(dataset_path, sep="|")
        self.X = self.df["sentence"]
        self.y = self.df["sentiment"] if "sentiment" in self.df.columns else None
        self.pipeline = None

        if verbose:
            cyan(f"[Data] Loaded {len(self.df)} records from {dataset_path}")
            if use_llm:
                yellow("[LLM] LLM features enabled.")
            if use_weak:
                yellow("[Weak] Weak classifier signals enabled.")

        self._check_and_label()

    def _check_and_label(self):
        llm_cols = ["sentiment_score", "confidence_rating", "explanation_score"]
        if self.use_llm and not all(col in self.df.columns for col in llm_cols):
            yellow("[LLM] Generating LLM features...")
            outputs = []
            valid_idx = []

            for idx, text in enumerate(self.X):
                result = label_row(text, verbose=self.verbose)
                if result:
                    outputs.append(result)
                    valid_idx.append(idx)
                elif self.verbose:
                    print(f"[LLM] Skipped row {idx}: LLM returned None")

            if not outputs:
                raise RuntimeError("LLM labeling failed: No valid outputs returned.")

            llm_values = list(zip(*outputs))
            for i, col in enumerate(llm_cols):
                self.df.loc[valid_idx, col] = llm_values[i]

            if self.df[llm_cols].isnull().any().any():
                yellow("[Warning] Some rows could not be labeled by the LLM.")

        if "vector" not in self.df.columns:
            if self.verbose:
                yellow("[Vectorizer] Generating text vectors for weak classifiers...")
            pre = TextPreprocessor(self.stop_words)
            vec = FeatureVectorizer()
            X_clean = pre.fit_transform(
                self.X
            )  # Fit + transform to match scikit-learn protocol
            vec.fit(X_clean)  # Fit the vectorizer
            X_vec = vec.transform(X_clean)  # Now safe to transform
            self.df["vector"] = list(
                X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
            )

        weak_cols = ["SVM", "NB", "LR", "RF"]
        if self.use_weak and not all(col in self.df.columns for col in weak_cols):
            yellow("[Weak] Generating weak classifier predictions...")
            self.df = apply_weak_classifiers(self.df)

    def _build_pipeline(self, X_input):
        from sklearn.pipeline import Pipeline as SkPipeline

        # Explicitly fit text components
        preprocess = TextPreprocessor(self.stop_words)
        vectorize = FeatureVectorizer()
        preprocess.fit(X_input["sentence"])
        X_clean = preprocess.transform(X_input["sentence"])
        vectorize.fit(X_clean)

        text_pipeline = SkPipeline(
            [
                ("preprocess", preprocess),
                ("vectorize", vectorize),
            ]
        )

        transformers = [("text", text_pipeline, "sentence")]

        if self.use_llm:
            transformers.append(
                (
                    "llm",
                    FunctionTransformer(extract_llm_features, validate=False),
                    ["sentiment_score", "confidence_rating", "explanation_score"],
                )
            )
        else:
            transformers.append(
                (
                    "noop_llm",
                    FunctionTransformer(noop_return_empty, validate=False),
                    ["sentence"],
                )
            )

        if self.use_weak:
            transformers.append(
                (
                    "weak",
                    FunctionTransformer(extract_weak_features, validate=False),
                    ["SVM", "NB", "LR", "RF"],
                )
            )
        else:
            transformers.append(
                (
                    "noop_weak",
                    FunctionTransformer(noop_return_empty, validate=False),
                    ["sentence"],
                )
            )

        column_fusion = ColumnTransformer(transformers, remainder="drop")
        column_fusion.fit(X_input)

        clf = XGBClassifier(eval_metric="logloss")

        self.pipeline = SkPipeline(
            [
                ("features", column_fusion),
                ("clf", clf),
            ]
        )

        return column_fusion, clf

    def train(self):
        cyan("→ Training model...")
        X_input = self._extract_features_df()

        column_fusion, clf = self._build_pipeline(X_input)
        X_transformed = column_fusion.transform(X_input)
        clf.fit(X_transformed, self.y)

        green("✓ Training complete.")
        self.debug_unfitted_components()

    def evaluate(self):
        cyan("→ Evaluating model on training set...")
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(self.pipeline)
        except Exception as e:
            yellow(f"[Warning] Pipeline not fully fitted: {e}")
            self.debug_unfitted_components()
            return 0.0

        X_input = self._extract_features_df()
        features = self.pipeline.named_steps["features"].transform(X_input)
        predictions = self.pipeline.named_steps["clf"].predict(features)
        acc = np.mean(predictions == self.y)
        green(f"✓ Accuracy on training set: {acc:.4f}")
        return acc

    def predict(self, sentence_list):
        self.debug_unfitted_components()
        temp_df = pd.DataFrame({"sentence": sentence_list})
        self.df = temp_df
        X_input = self._extract_features_df()
        features = self.pipeline.named_steps["features"].transform(X_input)
        predictions = self.pipeline.named_steps["clf"].predict(features)
        return predictions

    def save_model(self, path="trained_pipeline.pkl"):
        import joblib

        joblib.dump(self.pipeline, path)
        green(f"✓ Pipeline saved to {path}")

    def _extract_features_df(self):
        cols = ["sentence"]
        if self.use_llm:
            cols += ["sentiment_score", "confidence_rating", "explanation_score"]
        if self.use_weak:
            cols += ["SVM", "NB", "LR", "RF"]
        return self.df[cols]

    def debug_unfitted_components(self):
        from sklearn.utils.validation import check_is_fitted

        print("🔍 [Debug] Checking pipeline components for fit status...\n")

        def try_check(step_name, obj):
            try:
                check_is_fitted(obj)
                green(f"✅ Fitted: {step_name}")
            except Exception as e:
                red(f"❌ Not fitted: {step_name} — {e}")

        if hasattr(self.pipeline, "named_steps"):
            for name, step in self.pipeline.named_steps.items():
                if name == "features" and isinstance(step, ColumnTransformer):
                    for trans_name, transformer, _ in step.transformers:
                        try_check(f"[features → {trans_name}]", transformer)
                        if hasattr(transformer, "named_steps"):
                            for sub_name, sub_step in transformer.named_steps.items():
                                try_check(
                                    f"[features → {trans_name} → {sub_name}]", sub_step
                                )
                else:
                    try_check(name, step)
