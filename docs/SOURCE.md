# Source and packaging provenance

The release scope is **analysis code, derived results, and published baselines**. The current package contains the analysis functions and published review-policy implementations, and produces derived result files from user-supplied records. It does not distribute checkpoint binaries. The fixed models referenced in the methods are the models evaluated in the experiments.

The reusable package wraps the existing RV review evaluator. The scientific functions below are copied byte-for-byte from the preceding local candidate; no study results were recomputed for this packaging task.

| File | SHA-256 |
|---|---|
| `_locked_review.py` | `c6baed800d822654bf5bab6fea28b9698371b0f0b2f3cf4053426edb0076ccb5` |
| `review_adapter.py` | `5c35c4ccc9e9f821abe670bb78fe1459dd8a62e78596101790b812a8ca421075` |
| `propagation.py` | `fd565290c085242e761562e3a78d82c7b07d256a75d042e903ff200ff13c7cc0` |

The original source is identified by file hashes because the development workspace has no usable Git revision. The new repository has its own independent history. The public additions are package metadata, an installed CLI, a thin file-based Python API, and bundled synthetic example resources.

Dependencies preserve the previously tested numerical environment: NumPy 2.4.6, pandas 3.0.3, and PyYAML 6.0.3. Historical study replay and its patient data are not part of this repository.

The copied MIT licence is retained. This packaging task does not create new licence terms.

Package validation on Python 3.11.15: installation in a fresh virtual environment succeeded; 23 unit tests passed; the installed CLI ran the bundled synthetic example from outside the checkout (0.306 seconds inside the evaluator); output schemas and dependency consistency passed. No study analysis was rerun.
