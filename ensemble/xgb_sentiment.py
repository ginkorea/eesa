# xgb_sentiment.py

"""
High-performance XGBoost classifier for sentiment analysis with support for:
- LLM-based sentiment feature augmentation
- Weak classifier signal fusion
- Parallelized k-fold training
- Clean sklearn-style API
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional
from sklearn.model_selection import KFold
from util import cyan, green, yellow, red


class XGBSentimentClassifier:
    def __init__(
        self,
        include_llm: bool = False,
        include_weak: bool = False,
        n_splits: int = 5,
        max_depth: int = 3,
        eta: float = 0.1,
        multiproc: bool = False,
        random_state: int = 42,
        save_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.include_llm = include_llm
        self.include_weak = include_weak
        self.n_splits = n_splits
        self.max_depth = max_depth
        self.eta = eta
        self.multiproc = multiproc
        self.random_state = random_state
        self.save_dir = save_dir
        self.verbose = verbose

        self.results = []
        self.cv_results = []
        self.model_name = self._build_model_name()
        self.trained = False
        self.final_model = None  # ✅ will store the trained model

    def _build_model_name(self):
        name = "xgb"
        if self.include_llm:
            name += "_llm"
        if self.include_weak:
            name += "_weak"
        return f"{name}_depth{self.max_depth}"

    def _augment_vector(self, row):
        vec = row["vector"]
        if self.include_llm:
            llm = [row["sentiment_score"], row["confidence_rating"], row["explanation_score"]]
            vec = np.concatenate([vec, llm])
        if self.include_weak:
            weak = [row["SVM"], row["NB"], row["LR"], row["RF"]]
            vec = np.concatenate([vec, weak])
        return vec

    def _prepare_data(self, df: pd.DataFrame):
        df = df.copy()
        df["vector"] = df.apply(self._augment_vector, axis=1)
        df["fold"] = -1
        return df

    def _get_folds(self, df):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold_index, (_, test_idx) in enumerate(kf.split(df)):
            df.loc[test_idx, "fold"] = fold_index
        return df

    def fit(self, df: pd.DataFrame, num_rounds: int = 100):
        cyan("Preparing data...")
        df = self._prepare_data(df)
        df = self._get_folds(df)
        x_all = np.vstack(df["vector"].values)
        y_all = np.vstack(df["sentiment"].values).ravel()

        cyan("Starting training across folds...")
        self.cv_results = []

        for fold in range(self.n_splits):
            cyan(f"[Fold {fold}]")
            train_idx = df[df["fold"] != fold].index
            test_idx = df[df["fold"] == fold].index

            x_train, x_test = x_all[train_idx], x_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test, label=y_test)

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": self.max_depth,
                "eta": self.eta,
                "seed": self.random_state,
            }

            model = xgb.train(params, dtrain, num_boost_round=num_rounds)
            y_pred = model.predict(dtest)
            self.cv_results.append((fold, y_pred, test_idx))
            df.loc[test_idx, "results"] = y_pred
            df.loc[test_idx, "fold"] = fold

        self.processed_df = df

        # ✅ Final training on entire dataset for deployment use
        cyan("Training final model on all data...")
        dtrain_all = xgb.DMatrix(x_all, label=y_all)
        self.final_model = xgb.train(params, dtrain_all, num_boost_round=num_rounds)
        self.trained = True
        green("Training complete.")

        if self.save_dir:
            self._save_results()

    def _save_results(self):
        os.makedirs(self.save_dir, exist_ok=True)
        file_path = os.path.join(self.save_dir, f"{self.model_name}_cv_results.csv")
        self.processed_df.to_csv(file_path, index=False, sep="|")
        if self.verbose:
            green(f"Saved CV results to {file_path}")

    def save_model(self, path="models/xgb_model.bin"):
        """Save the final trained model."""
        if not self.final_model:
            raise RuntimeError("No model to save. Train the model first.")
        self.final_model.save_model(path)
        green(f"✓ Final model saved to {path}")

    def load_model(self, path="models/xgb_model.bin"):
        """Load a previously trained model."""
        self.final_model = xgb.Booster()
        self.final_model.load_model(path)
        self.trained = True
        green(f"✓ Model loaded from {path}")

    def evaluate(self):
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        y_true = np.vstack(self.processed_df["sentiment"].values).ravel()
        y_pred = self.processed_df["results"].values
        binary_pred = (y_pred > 0.5).astype(int)
        acc = np.mean(binary_pred == y_true)
        log_loss = -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))
        return acc, log_loss

    def predict(self, df: pd.DataFrame):
        if not self.trained or self.final_model is None:
            raise RuntimeError("No trained model available. Train or load the model first.")
        df = df.copy()
        df["vector"] = df.apply(self._augment_vector, axis=1)
        x = np.vstack(df["vector"].values)
        dmatrix = xgb.DMatrix(x)
        return self.final_model.predict(dmatrix)

# === EXPORTABLE UTILS FOR OTHER MODULES ===

def include_llm_vector(row, include_weak=False, verbose=False):
    """
    Appends LLM sentiment scores and optionally weak classifier predictions to the vector.

    Args:
        row (pd.Series): A row with 'vector' and LLM/weak classifier fields.
        include_weak (bool): Whether to include weak classifiers as features.
        verbose (bool): Print debug information.

    Returns:
        np.ndarray: Extended feature vector.
    """
    base_vector = row["vector"]
    if "sentiment_score" in row and "confidence_rating" in row and "explanation_score" in row:
        base_vector = np.concatenate([
            base_vector,
            [row["sentiment_score"], row["confidence_rating"], row["explanation_score"]]
        ])
    if include_weak:
        weak = [row.get("SVM", 0), row.get("NB", 0), row.get("LR", 0), row.get("RF", 0)]
        base_vector = np.concatenate([base_vector, weak])
    if verbose:
        cyan(f"Vector augmented → {base_vector[-5:]}")
    return base_vector
