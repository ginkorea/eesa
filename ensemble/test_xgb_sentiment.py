import numpy as np
import pandas as pd
from ensemble.xgb_sentiment import XGBSentimentClassifier, include_llm_vector


def make_dummy_data(n=50, d=5):
    """Generates dummy dataframe with required fields."""
    np.random.seed(42)
    df = pd.DataFrame({
        "vector": [np.random.rand(d) for _ in range(n)],
        "sentiment": np.random.randint(0, 2, size=(n, 1)).tolist(),
        "sentiment_score": np.random.uniform(-1, 1, size=n),
        "confidence_rating": np.random.uniform(0, 1, size=n),
        "explanation_score": np.random.uniform(0, 1, size=n),
        "SVM": np.random.randint(0, 2, size=n),
        "NB": np.random.randint(0, 2, size=n),
        "LR": np.random.randint(0, 2, size=n),
        "RF": np.random.randint(0, 2, size=n),
    })
    return df


def test_include_llm_vector():
    df = make_dummy_data(1)
    row = df.iloc[0]
    out = include_llm_vector(row, include_weak=True)
    expected_len = len(row["vector"]) + 3 + 4  # vector + llm + weak
    assert isinstance(out, np.ndarray)
    assert len(out) == expected_len
    print("✓ include_llm_vector passed.")


def test_training_and_evaluation():
    df = make_dummy_data()
    clf = XGBSentimentClassifier(
        include_llm=True,
        include_weak=True,
        n_splits=5,
        max_depth=3,
        eta=0.2,
        save_dir=None  # skip saving
    )
    clf.fit(df, num_rounds=5)
    assert clf.trained
    assert "results" in clf.processed_df.columns
    acc, loss = clf.evaluate()
    assert 0 <= acc <= 1
    assert loss > 0
    print("✓ XGBSentimentClassifier training + evaluation passed.")


def test_fold_split():
    df = make_dummy_data()
    clf = XGBSentimentClassifier()
    df = clf._get_folds(df)
    assert set(df["fold"].unique()) == set(range(clf.n_splits))
    print("✓ Fold splitting passed.")


if __name__ == "__main__":
    print("=== Running XGBSentimentClassifier Tests ===")
    test_include_llm_vector()
    test_fold_split()
    test_training_and_evaluation()
    print("✓ All tests passed.")
