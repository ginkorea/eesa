# eesa.py
# CLI for EESA: Ensemble-based Sentiment Analysis

import argparse
import pandas as pd
import joblib
from pipeline import SentimentPipeline
from ensemble import run_comparison_on_dataset
from util import cyan, green


def train(
    file: str,
    name: str,
    save_dir: str = "results",
    verbose: bool = False,
    llm: bool = False,
    weak: bool = False,
):
    cyan(f"Starting training on {file}")
    pipe = SentimentPipeline(file, use_llm=llm, use_weak=weak, verbose=verbose)
    pipe.train()
    pipe.evaluate()
    pipe.save_model(f"{save_dir}/{name}_pipeline.pkl")


def label(file: str, batch_size: int, batch_start: int):
    pipe = SentimentPipeline(file, use_llm=True, verbose=True)
    pipe.label_and_save(start=batch_start, batch_size=batch_size)


def compare(file: str):
    df = pd.read_csv(file, sep="|")
    run_comparison_on_dataset(df)


def infer(model_path: str, input_path: str, output_path: str = None):
    model = joblib.load(model_path)
    df = pd.read_csv(input_path, sep="|")
    X = df["sentence"].tolist()
    preds = model.predict(X)
    df["prediction"] = preds
    if output_path:
        df.to_csv(output_path, sep="|", index=False)
        green(f"✓ Saved to {output_path}")
    else:
        print(df[["sentence", "prediction"]].head())


def parse_args():
    parser = argparse.ArgumentParser(description="EESA: Sentiment CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_cmd = subparsers.add_parser("train")
    train_cmd.add_argument("file")
    train_cmd.add_argument("name")
    train_cmd.add_argument("--save_dir", default="results")
    train_cmd.add_argument("--verbose", action="store_true")
    train_cmd.add_argument("--llm", action="store_true")
    train_cmd.add_argument("--weak", action="store_true")

    label_cmd = subparsers.add_parser("label")
    label_cmd.add_argument("file")
    label_cmd.add_argument("batch_size", type=int)
    label_cmd.add_argument("batch_start", type=int)

    compare_cmd = subparsers.add_parser("compare")
    compare_cmd.add_argument("file")

    infer_cmd = subparsers.add_parser("infer")
    infer_cmd.add_argument("model_path")
    infer_cmd.add_argument("input_path")
    infer_cmd.add_argument("--output_path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        train(
            file=args.file,
            name=args.name,
            save_dir=args.save_dir,
            verbose=args.verbose,
            llm=args.llm,
            weak=args.weak,
        )
    elif args.command == "label":
        label(file=args.file, batch_size=args.batch_size, batch_start=args.batch_start)
    elif args.command == "compare":
        compare(args.file)
    elif args.command == "infer":
        infer(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
        )
    else:
        print("No command provided. Use --help for options.")
