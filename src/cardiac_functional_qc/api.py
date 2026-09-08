"""Public file-based API around the existing, unchanged review evaluator."""
from importlib.resources import files
from pathlib import Path

from .review_adapter import run


def evaluate_review(input_csv, config, output_dir):
    """Evaluate labelled RV records and return the output directory as a Path.

    ``input_csv`` is a CSV of predictions, references, and existing QC scores.
    ``config`` is an explicit YAML configuration (see ``write_example``).
    ``output_dir`` must be new or empty. Validation errors are raised and logged.
    This is an external-input evaluation; it never reads frozen paper results.
    """
    output = Path(output_dir).resolve()
    run(Path(input_csv), Path(config), output, paper=False)
    return output


def write_example(output_dir):
    """Write a synthetic input CSV and matching configuration; return their paths.

    Existing files are never overwritten. No evaluation runs in this function.
    """
    destination = Path(output_dir).resolve()
    paths = (destination / "review_input.csv", destination / "review.yaml")
    if any(p.exists() for p in paths):
        raise FileExistsError("Example files already exist; choose a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    resources = files("cardiac_functional_qc").joinpath("resources")
    for target in paths:
        target.write_bytes(resources.joinpath(target.name).read_bytes())
    return paths
