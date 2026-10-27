QC TRANSFER GAP: ANALYSIS SCRIPTS

Install the repository with python -m pip install . . The scripts import the
budget-matched review package in src/.

PRIVATE INPUT CONTRACT
Supply your own CSV, with one row per cohort, patient_id, architecture and
ventricle. Required columns are cohort, patient_id, architecture, ventricle,
pred_edv_ml, pred_esv_ml, ref_edv_ml, ref_esv_ml, pred_ef, ref_ef and dice.
EF is in percentage points and volumes are mL. dice is the arithmetic mean
of the ED and ES processed-grid 3D Dice values. Native volumes use the NIfTI
voxel volume. Optional stored EDV, SV and EF absolute-error columns and SV
volumes are retained, after consistency checks, to preserve the exact saved
floating-point inputs. Otherwise they are derived from the measurement columns.

The four scores are softmax_entropy,
measurement_aware_softmax_entropy, mc_dropout and
measurement_aware_mc_dropout. Scores are already patient-level means over
the two phases. Larger scores mean higher review priority. Use ACDC and MNMS
as cohort names, and RV or LV as ventricle names. The four fixed architecture
names are unet, segresnet, attnunet and swinunetr.

No private input records, IDs or split lists accompany these sources.
analysis.prepare_inputs converts the original separate RV/LV patient/QC tables.
For M&Ms subgroup analyses, join the official vendor code into a vendor column;
analysis.vendor can perform that join with --metadata. A=Siemens, B=Philips,
C=GE, D=Canon. No case-to-vendor mapping is embedded in the source.

RUN FROM THE REPOSITORY ROOT
python -m analysis.reference --input private/input.csv --output outputs/reference
python -m analysis.recovered_fraction --input private/input.csv --output outputs/RF
python -m analysis.partial_intervals --input private/input.csv --output outputs/partial
python -m analysis.relative_errors --input private/input.csv --output outputs/relative
python -m analysis.vendor --input private/input_with_vendor.csv --output outputs/vendor
python -m analysis.tolerance --input private/input.csv --output outputs/tolerance
python -m analysis.single_grid --input private/input.csv --phase-table private/phase_grid.csv --output outputs/grids
python -m analysis.disagreement --input private/input.csv --output outputs/disagreement
python -m analysis.mc_spread --deterministic-input private/input_with_vendor.csv --passes private/MC_passes.csv --entropy-input private/new_MC_scores.csv --output outputs/MC

--draws defaults to 5000. --draws 0 exports point estimates without intervals.
The ordinary architecture median always uses all supplied models, not a median
of only defined models. Table S6 uses the four fixed architectures; supply only
those four for that table. Its partial-correlation range is across the four
score medians, after taking each score's architecture median. It is not an
architecture range. RF is the un-clipped discrete Eq.13 recovered fraction.

BOOTSTRAP, TIES AND SEEDS
Rank ties use average ranks, recomputed inside each resample. The rank and
partial kernels use fixed per-architecture CRC32 seeds. The joint RF
bootstrap uses a fixed cell seed, shared across models and policies;
stable patient-ID ordering precedes the original draw occurrence order, then
the unchanged stable descending-score kernel. AURCs for policy, random and
true-error oracle are recomputed inside each draw. Undefined RF draws use the
paper machine-epsilon guard and are counted, not stabilised or clipped.
Partial architecture-median intervals use the same joint cell seeds as the
Dice-oracle intervals. The vendor, tolerance and spread analyses use fixed
shared cohort draws and vendor seed recipes. Seed ledgers are generated at runtime.

SENSITIVITIES WITHOUT COMMITTED IDS
--sensitivity exclude_pathological_10_rows requires the private frozen
pathological_row boolean column, shared across ventricles for the four fixed
models. --sensitivity exclude_reference_3 requires a private exclude_reference
boolean column and removes flagged rows on MNMS only. No ID lists are stored.
The nnU-Net baseline's pathological-ratio flags must be computed from its
own predictions;
do not reuse another model's flagged rows. Relative SV correlations exclude
reference SV <= 0 and undefined ratios only; an exclusion-count file is written.

GRIDS AND SAVED MASKS
analysis.phase_grid_measurements accepts a private manifest with cohort,
patient_id, architecture, phase, image, reference and processed_prediction.
Use an optional frame column for 4D data. Native exported predictions may
instead be supplied in native_prediction, especially for nnU-Net.
The result supplies analysis.single_grid. M&Ms references are remapped from
0,1,2,3 to 0,3,2,1. Native reconstruction restores centre crop/pad geometry,
fills native pixels outside a crop with background, and resizes hard labels
with nearest neighbour. It does not reconstruct cropped-away anatomy.
Score stride two means the fixed [::2,::2] in-plane decimation, without z
decimation. The single-grid analysis keeps the original scores fixed.

SPREAD SCORES
MC input columns are cohort, architecture, patient_id, ventricle,
pass_ordinal, phase and native_volume_ml. ED and ES are paired by pass ordinal.
There must be twenty passes. EF/EDV/SV SD uses ddof=0; EF IQR uses linear
quartiles. Direct entropy comparisons use new_mc_dropout and
new_measurement_aware_mc_dropout from the SAME new passes, not stored old MC
entropy. The deterministic deployed measurements remain the error targets.
The same statistic kernel and shared draws are used for the spread scores, including paired RF(Dice)-RF(EF-SD), paired new-entropy comparisons,
and Siemens/non-Siemens. The numerical four-model mean uses arithmetic means
of the four EF/EDV/SV predictions. Its Dice-dependent analyses are undefined.
For the nnU-Net fold spread, pass --fold-input to analysis.disagreement. That private
table supplies cohort, ventricle, patient_id, fold, pred_edv_ml, pred_esv_ml
and pred_ef. The deployed nnU-Net input must be the native probability ensemble.

TOLERANCE
Review counts are floor(budget*n), with prediction-only plausibility flags
consuming the same exact budget. Random policies are analytical expectations.
The required-budget search checks every nonempty released set; infinity means
not attained before full review and remains in its percentile distribution.
The 10% trimmed mean removes floor(0.10*n_retained) from each tail.

PORTABILITY
The command-line and data-loading adapters are portable wrappers around the
mathematical kernels and were validated against the saved study estimates.
