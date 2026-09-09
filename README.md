# QC Transfer Gap

Evaluate whether a segmentation quality-control score helps prioritize review of the **RV ejection-fraction measurements** derived from those segmentations.

Supply patient-level predictions, reference measurements, and your existing QC scores. The package compares five review policies at the same workload and writes retained-error estimates, paired bootstrap intervals, and review allocations. It uses the existing study evaluator with a small Python API and command-line interface. No medical images, model weights, PyTorch, or GPU are needed.

The release scope is **analysis code, derived results, and published baselines**. This package provides the analysis code and implementations of the published review-policy baselines; derived results are generated from the records supplied to the evaluator, as described under [Outputs](#outputs). **Checkpoint binaries for the fixed models evaluated in the study are not distributed.** References to those fixed models describe the experimental evaluation.

## Install

Use Python **3.11 or later**. From a checkout:

```bash
python -m pip install .
```

Or install directly from GitHub (repository access is required while it is private):

```bash
python -m pip install "git+https://github.com/gionanide/qc-transfer-gap.git"
```

## Try the synthetic example

These commands work from any directory after installation:

```bash
qc-transfer-gap init --output my-review
qc-transfer-gap evaluate --input my-review/review_input.csv --config my-review/review.yaml --output my-review/results
```

The ten example records are synthetic and demonstrate the interface only. `init` writes the CSV and configuration so you can inspect and edit both. Use a new output directory for each evaluation.

If the executable is not on your PATH, use `python -m cardiac_functional_qc` with the same arguments.

## Use your own data

Replace the example CSV with one row per patient and model. Volumes are in **mL**; EF values are in **percentage points** (60 means 60%).

| Column | Meaning |
|---|---|
| `cohort` | Dataset/group identifier; cohorts are evaluated separately |
| `patient_id` | Stable patient identifier; preserve leading zeros |
| `architecture` | Your model identifier; a single model is supported |
| `ventricle` | `RV` (or omit when the configuration declares RV) |
| `pred_edv_ml`, `pred_esv_ml` | Predicted end-diastolic/end-systolic volumes |
| `pred_rvef_points` | Predicted EF, consistent with those volumes |
| `ref_rvef_points` | Reference EF; reference EDV and ESV can be supplied instead |
| Your QC-score column(s) | Numeric scores, with larger values meaning higher review priority |

Edit `scores` in `review.yaml` to name your score columns, and replace `SYNTHETIC` under `cohort_seeds` with your cohort name and a fixed integer seed. Add an entry for every cohort. Models within a cohort must have identical patient membership and reference values. The package rejects inconsistent or unsupported input with an explicit error.

```bash
qc-transfer-gap evaluate --input my_patient_results.csv --config review.yaml --output results/my_review
```

See [the complete input contract](docs/INPUT_CONTRACT.md) for optional fields, flags, ties, and invalid-value handling. This is a labelled audit of existing scores: the package does not generate segmentation predictions or QC scores.

## Python API

```python
from cardiac_functional_qc import evaluate_review, write_example

input_csv, config = write_example("my-review")
output = evaluate_review(input_csv, config, "my-review/results")

import pandas as pd
policies = pd.read_csv(output / "policy_risks.csv")
print(policies[["policy", "estimate", "ci_low", "ci_high"]])
```

Pass your own CSV and YAML paths to the same `evaluate_review` function.

## What it evaluates

The preserved protocol uses 10% and 20% review budgets and 5,000 paired patient bootstrap draws. The five policies are random review, QC alone, plausibility plus random, plausibility plus QC, and a reference-error oracle. The oracle is an audit comparator requiring ground truth.

Retained risk is mean absolute RV EF error among the unreviewed patients. Flags consume places within the same budget. Selection uses predictions and QC scores; references supply audit errors and the oracle. See [method details](docs/METHOD.md).

## Outputs

| File | Contents |
|---|---|
| `policy_risks.csv` | Five policies with retained-error estimates and intervals |
| `architecture_results.csv` | Per-model policy risks and paired contrasts |
| `main_results.csv` | Summaries across the models actually supplied |
| `flag_capacity.csv`, `plausibility_audit.csv` | Flags and review-capacity accounting |
| `patient_review_allocations.csv` | Deterministic selections or analytical review probabilities |
| `bootstrap_metrics_and_indices.npz` | Paired resampling audit data |
| `run_manifest.json` | Configuration/input hashes, versions, and runtime |

The inherited evaluator also writes a Markdown summary named `Table4.md`. Output files remain in the directory you specify.

## Development

```bash
python -m pip install ".[test]"
python -m pytest
```

Tests exercise policy arithmetic, malformed inputs, and the installed interface using synthetic data. The repository contains reusable code and examples; study patient records and paper-result reproduction are outside this package. Existing evaluation implementation provenance is recorded in [SOURCE.md](docs/SOURCE.md).
