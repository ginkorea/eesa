# pipeline_utils.py

import numpy as np


def extract_llm_features(df):
    return df[["sentiment_score", "confidence_rating", "explanation_score"]].values


def extract_weak_features(df):
    return df[["SVM", "NB", "LR", "RF"]].values


def noop_return_empty(X):
    return np.empty((len(X), 0))
