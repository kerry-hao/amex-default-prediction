"""Command-line entry point; each stage runs only when explicitly selected."""

import argparse
import json

from .config import load_config, plan


def main():
    parser = argparse.ArgumentParser(description="American Express default prediction workflow")
    parser.add_argument("--config", default="configs/baseline.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Show workflow and paths")
    prepare_parser = sub.add_parser("prepare", help="Aggregate raw statements to customer features")
    prepare_parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    prepare_parser.add_argument("--overwrite", action="store_true")
    train_parser = sub.add_parser("train", help="Fit all folds and write OOF results")
    train_parser.add_argument("--overwrite", action="store_true")
    sub.add_parser("predict", help="Create submission from trained folds")
    sub.add_parser("report", help="Create OOF figures and results markdown")
    sub.add_parser("profile", help="Summarize prepared customer data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "plan":
        result = plan(cfg)
    elif args.command == "prepare":
        from .features import prepare
        splits = ["train", "test"] if args.split == "both" else [args.split]
        result = [prepare(cfg, split, args.overwrite) for split in splits]
    elif args.command == "train":
        from .training import train
        result = train(cfg, args.overwrite)
    elif args.command == "predict":
        from .prediction import predict
        result = predict(cfg)
    elif args.command == "report":
        from .reporting import report
        result = report(cfg)
    else:
        from .reporting import profile
        result = profile(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
