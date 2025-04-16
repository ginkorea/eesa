# eesa.py

import argparse
import time
import pandas as pd
import joblib
from pipeline import SentimentPipeline
from ensemble import XGBSentimentClassifier, run_comparison_on_dataset
from util import cyan, green, red


def train(file: str, name: str, depth: int = 3, include_llm: bool = False,
          include_weak: bool = False, save_dir: str = "results"):
    """Trains an XGBoost model and a LogisticRegression pipeline."""
    start = time.time()
    cyan(f"Starting pipeline on {file}")
    pipe = SentimentPipeline(file, use_llm=include_llm)
    pipe.train()
    pipe.evaluate()
    pipe.save_model(f"{save_dir}/{name}_pipeline.pkl")

    clf = XGBSentimentClassifier(
        include_llm=include_llm,
        include_weak=include_weak,
        max_depth=depth,
        save_dir=save_dir
    )
    clf.fit(pipe.df)
    clf.evaluate()
    clf.save_model(f"{save_dir}/{name}_xgb_model.pkl")
    green(f"Finished training {name} in {time.time() - start:.2f}s")


def label(file: str, batch_size: int, batch_start: int):
    """Performs batch LLM labeling and saves outputs."""
    start = time.time()
    cyan(f"Labeling {file} from {batch_start} to {batch_start + batch_size}")
    pipe = SentimentPipeline(file, use_llm=True)
    pipe.label_and_save(start=batch_start, batch_size=batch_size)
    green(f"Finished LLM labeling in {time.time() - start:.2f}s")


def compare(file: str):
    """Performs ANOVA-based comparison of weak classifiers."""
    df = pd.read_csv(file, sep="|")
    run_comparison_on_dataset(df)


def infer(model_path: str, input_path: str, output_path: str = None, use_llm: bool = False):
    """Runs inference using a trained pipeline model."""
    green(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    df = pd.read_csv(input_path, sep="|") if input_path.endswith(".csv") else pd.read_csv(input_path)
    if "sentence" not in df.columns:
        raise ValueError("Input file must contain a 'sentence' column.")

    X = df["sentence"].tolist()
    cyan(f"Running predictions on {len(X)} samples...")

    try:
        y_pred = model.predict(X)
    except Exception as e:
        red(f"Prediction error: {e}")
        return

    df["prediction"] = y_pred

    if output_path:
        df.to_csv(output_path, sep="|", index=False)
        green(f"Saved predictions to {output_path}")
    else:
        print(df[["sentence", "prediction"]].head())


def parse_args():
    parser = argparse.ArgumentParser(description="EESA: Ensemble-based Explainable Sentiment Analysis CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train XGBoost + Pipeline model")
    train_parser.add_argument("file", help="Path to input CSV")
    train_parser.add_argument("name", help="Model name identifier")
    train_parser.add_argument("--llm", action="store_true", help="Include LLM sentiment features")
    train_parser.add_argument("--weak", action="store_true", help="Include weak classifier features")
    train_parser.add_argument("--depth", type=int, default=3, help="XGBoost tree depth")
    train_parser.add_argument("--shrink", action="store_true", help="Shrink dataset for faster training")
    train_parser.add_argument("--save_dir", default="results", help="Directory to save model/results")

    # Label
    label_parser = subparsers.add_parser("label", help="Batch label using LLM")
    label_parser.add_argument("file", help="Path to CSV file")
    label_parser.add_argument("batch_size", type=int, help="Batch size")
    label_parser.add_argument("batch_start", type=int, help="Batch start index")

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Run weak classifier ANOVA comparison")
    compare_parser.add_argument("file", help="Path to labeled dataset")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference using trained model")
    infer_parser.add_argument("model_path", help="Path to trained .pkl model")
    infer_parser.add_argument("input_path", help="CSV with 'sentence' column")
    infer_parser.add_argument("--output_path", help="Optional CSV to write predictions")
    infer_parser.add_argument("--llm", action="store_true", help="Model includes LLM inputs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        train(
            file=args.file,
            name=args.name,
            depth=args.depth,
            include_llm=args.llm,
            include_weak=args.weak,
            save_dir=args.save_dir
        )
    elif args.command == "label":
        label(
            file=args.file,
            batch_size=args.batch_size,
            batch_start=args.batch_start
        )
    elif args.command == "compare":
        compare(args.file)
    elif args.command == "infer":
        infer(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
            use_llm=args.llm
        )
    else:
        print("No command provided. Use --help for options.")
