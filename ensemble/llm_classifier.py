# llm_classifier.py

"""
Ensemble-based LLM sentiment classifier using OpenAI GPT models.
Provides scoring, confidence, and explanation with summarization.
"""

import re
import numpy as np
import pandas as pd
from openai_llm.sentiment import SentiChat, SentiSummary
from util import red, green, cyan


class LLMClassifier:
    """
    Ensemble-based LLM classifier that runs multiple sentiment scorers and averages their outputs.

    Args:
        num_classifiers (int): Number of parallel GPT classifiers to use (voting ensemble)
        verbose (bool): Whether to print debug information
    """

    def __init__(self, num_classifiers: int = 3, verbose: bool = False):
        self.verbose = verbose
        self.num_classifiers = num_classifiers
        self.classifiers = [SentiChat(debug=verbose) for _ in range(num_classifiers)]
        self.results = []

    def classify(self, text: str) -> list[str]:
        """
        Collects results from all classifiers for a given input text.

        Args:
            text (str): Input sentence for analysis

        Returns:
            list[str]: Raw GPT responses from each classifier
        """
        self.results = [
            clf.classify_sentiment(text, verbose=self.verbose)
            for clf in self.classifiers
        ]
        return self.results

    def parse_results(self) -> list[list]:
        """
        Splits each result into its 4 components: sentiment score, confidence, explanation quality, explanation.

        Returns:
            list[list]: Structured data
        """
        parsed = []
        for result in self.results:
            parts = result.split("|")
            parsed.append(
                [self._to_float(parts[i]) if i < 3 else parts[i] for i in range(4)]
            )
        return parsed

    def summarize_explanations(self, explanations: list[str]) -> str:
        """
        Uses an LLM summarizer to create a unified explanation from multiple GPT outputs.

        Args:
            explanations (list[str]): List of explanation texts

        Returns:
            str: Summarized reasoning
        """
        summarizer = SentiSummary()
        return summarizer.summarize_explanations(explanations, verbose=self.verbose)

    @staticmethod
    def _to_float(val):
        """Helper to extract float from string."""
        try:
            match = re.findall(r"-?\d+\.\d+|-?\d+", val)
            return float(match[0]) if match else np.nan
        except ValueError as e:
            red(f"Conversion error: {e}")
            return np.nan

    def average_results(self, parsed: list[list]) -> list:
        """
        Averages the numerical outputs and summarizes the explanations.

        Args:
            parsed (list[list]): Parsed results [[score, confidence, quality, explanation], ...]

        Returns:
            list: [avg_score, avg_confidence, avg_quality, explanation_summary]
        """
        df = pd.DataFrame(
            parsed,
            columns=[
                "sentiment_score",
                "confidence_rating",
                "explanation_score",
                "explanation",
            ],
        )
        try:
            summary = self.summarize_explanations(df["explanation"].tolist())
        except Exception as e:
            red(f"Summarization error: {e}")
            summary = df["explanation"].iloc[0] if not df["explanation"].empty else ""

        return [
            df["sentiment_score"].mean(),
            df["confidence_rating"].mean(),
            df["explanation_score"].mean(),
            summary,
        ]

    def classify_text(self, text: str) -> list:
        """
        Full pipeline for analyzing sentiment using an LLM ensemble.

        Args:
            text (str): Input sentence

        Returns:
            list: [sentiment_score, confidence_rating, explanation_score, explanation]
        """
        self.classify(text)
        parsed = self.parse_results()
        return self.average_results(parsed)


# === EXPORTABLE UTILS FOR OTHER MODULES ===


def label_row(
    text: str, existing_row: dict = None, verbose: bool = False
) -> list | None:
    if not text or not isinstance(text, str) or text.strip() == "":
        if verbose:
            print(f"[LLM] Skipped invalid text.")
        return None

    if existing_row and all(
        k in existing_row and existing_row[k] is not None
        for k in ["sentiment_score", "confidence_rating", "explanation_score"]
    ):
        return [
            existing_row["sentiment_score"],
            existing_row["confidence_rating"],
            existing_row["explanation_score"],
            existing_row.get("explanation", ""),
        ]

    try:
        classifier = LLMClassifier(verbose=verbose)
        return classifier.classify_text(text)
    except Exception as e:
        if verbose:
            print(f"[LLM] Error during labeling: {e}")
        return None
