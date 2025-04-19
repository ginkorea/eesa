import numpy as np
import pandas as pd
from ensemble.anova_analysis import run_comparison_on_dataset, ClassifierComparison, perform_anova
from ensemble.weak_models import (
    SVMClassifier,
    NaiveBayesClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifierWrapper
)


from sklearn.model_selection import KFold

def make_dummy_data(n=40, d=8, folds=5):
    np.random.seed(0)
    df = pd.DataFrame({
        "vector": [np.random.rand(d) for _ in range(n)],
        "sentiment": np.random.randint(0, 2, size=(n, 1)).tolist()
    })
    # Add a "fold" column
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_indices = np.zeros(n, dtype=int)
    for i, (_, test_idx) in enumerate(kf.split(df)):
        fold_indices[test_idx] = i
    df["fold"] = fold_indices
    return df



def test_perform_anova_structure():
    print("✓ perform_anova structure validated.")
    # Simulated predictions from 3 classifiers on same dataset
    pred_sets = [
        np.array([0, 1, 1, 0, 1]),
        np.array([1, 1, 1, 0, 1]),
        np.array([0, 0, 1, 0, 1])
    ]
    f_scores, p_values = perform_anova(pred_sets)
    assert isinstance(f_scores, list) and isinstance(p_values, list)
    assert len(f_scores) == 1 and len(p_values) == 1



def test_classifier_comparison():
    df = make_dummy_data()
    classifiers = {
        "SVM": SVMClassifier(df),
        "NB": NaiveBayesClassifier(df),
        "LR": LogisticRegressionClassifier(df),
        "RF": RandomForestClassifierWrapper(df)
    }
    comp = ClassifierComparison(classifiers, df)
    comp.compare()
    for k, v in comp.results.items():
        assert "accuracy_scores" in v
        assert "f_scores" in v
        assert "p_values" in v
        assert len(v["accuracy_scores"]) == comp.k
    print("✓ ClassifierComparison logic validated.")


def test_run_comparison_flow():
    df = make_dummy_data()
    run_comparison_on_dataset(df)
    print("✓ Full ANOVA comparison test ran successfully.")


if __name__ == "__main__":
    print("=== Running ANOVA Analysis Tests ===")
    test_perform_anova_structure()
    test_classifier_comparison()
    test_run_comparison_flow()
    print("✓ All ANOVA tests passed.")
