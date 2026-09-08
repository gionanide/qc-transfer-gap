"""Command-line interface installed by pip."""
import argparse
import sys
from . import __version__
from .api import evaluate_review, write_example


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate QC review policies from labelled RV measurement records.")
    parser.add_argument("--version", action="version", version=f"qc-transfer-gap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write a synthetic CSV and editable YAML configuration")
    init.add_argument("--output", required=True, help="Directory for the two example files")
    evaluate = commands.add_parser("evaluate", help="Evaluate your CSV using an explicit configuration")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output", required=True, help="New or empty directory for results")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            for path in write_example(args.output):
                print(path)
        else:
            print(f"Results: {evaluate_review(args.input, args.config, args.output)}")
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
