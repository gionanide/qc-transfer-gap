# RV review input contract

One row per `(cohort, patient_id, architecture, ventricle)`. String identifiers are preserved, including leading zeros. `architecture` is the native model-ID field; it need not describe a network family. Every model in a cohort must have exactly the same patients and reference measurements. Cohorts remain separate. Duplicate columns, duplicate keys, missing records, mixed ventricles, inconsistent references, invalid units, inconsistent supplied flags, and nonfinite numeric values produce explicit errors. No intersection, deduplication, clipping or hidden penalty occurs.

| Fields | Contract |
|---|---|
| `cohort`, `patient_id`, `architecture` | Nonempty stable string IDs. |
| `ventricle` | `RV`; may be omitted only because the YAML explicitly declares this RV-only dataset. |
| `pred_edv_ml`, `pred_esv_ml`, `pred_rvef_points` | Native-volume-derived predicted EDV/ESV in mL and EF in percentage points, consistent to absolute tolerance 1e-10. |
| `ref_rvef_points` | Reference EF in points, or supply both `ref_edv_ml` and `ref_esv_ml`; the existing `propagation.ef_points` computes reference EF. If both are supplied, consistency is checked. |
| Configured score columns | Finite numeric values; stable column identifiers declared in YAML; `direction: higher`. There is no test-set direction learning. |
| Existing optional fields | Reference EDV/ESV/SV, predicted SV, signed/absolute EF error, phase Dice and arithmetic ED/ES Dice mean can be retained. If supplied, numeric finiteness, EF error and Dice-mean consistency are checked. |
| Stored flags | Optional explicit true/false or 1/0 columns with the names below. If supplied, each must match recomputation from predictions. |

Flags retain the original definitions: `predicted_edv_le_zero`, `predicted_esv_gt_predicted_edv`, `predicted_ef_outside_0_100`, `nonfinite_predicted_ef`, and their union `pathological_ratio_any`. Reasons and ratio severity are emitted in `plausibility_audit.csv`. Finite negative EF or EF above 100 is retained in the starting population. The v1 interface requires defined, finite measurements and scores; undefined EF/nonpositive EDV cases are audited and rejected rather than assigned a finite error. The historical kernel includes nonfinite priority handling, but this release does not claim support for a new nonfinite-error eligibility convention.

QC sorting is descending numeric score, then ascending patient ID. Flags use the locked overflow order: among flagged records, nonfinite predicted EDV/ESV/EF or nonpositive EDV has priority; remaining finite violations sort by decreasing `max(0,-ESV/EDV,ESV/EDV-1)`, then patient ID. Only predicted quantities determine flag order. Repeated bootstrap copies keep their integer multiplicity; when a tie straddles capacity, partial selection of identical copies is valid and leaves the same risk.

Five thousand PCG64 patient draws are shared across all models, scores, policies and budgets in each cohort. External input requires an explicit fixed seed for each cohort. Reference values affect labelled audit errors and oracle rankings; they do not determine QC or plausibility ordering. External single-model input remains one model in every summary.

The paper's scores use two base signals: four-class softmax entropy and predictive entropy of the mean probabilities from 20 MC-dropout passes. Each has a plain and a measurement-aware aggregation, making four scores total. Plain RV aggregation uses predicted RV support; measurement-aware aggregation uses two dilation iterations and predicted slice-area weighting. ED/ES scores are arithmetically averaged. The independent LV replication used **LV-specific predicted regions and LV slice-area weights**, with the same formulas and checkpoints; it did not reuse RV patient scores. This package evaluates RV EF only.

CSV exports use 17 significant digits and the round-trip float parser. Preserve the available precision in your inputs. Schema/config path values are resolved relative to the config file; custom input/output command arguments are resolved from the working directory. The installed interface requires no author-machine paths.
