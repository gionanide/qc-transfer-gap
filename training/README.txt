FOUR FIXED MODELS

train.py is the preserved original U-Net loop; train_multi.py supports all four
architectures. model.py/model_zoo.py preserve the exact MONAI constructors,
dropout p=0.1 and placement. The training schedule is 100 epochs, Adam at
1e-3, batch size 16, DiceCE loss, cosine scheduling and two independent flips.
The model config JSON files describe these constructor settings; the original
constructors in model_zoo.py are authoritative.

Set QC_DATA_ROOT to the local ACDC database folder and QC_TRAIN_OUTPUT to a new
output folder. No machine-specific path is compiled into the sources.
python training/train.py
python training/train_multi.py --arch segresnet
python training/train_multi.py --arch attnunet
python training/train_multi.py --arch swinunetr --amp

The preserved source config and recovered original U-Net console text use
VAL_FRACTION=0.15 and split seed 42. The historical split file was not saved.
Table S11 uses the separately requested reconstructed 80/20 split with seed
42, not a recovered historical membership list. The reconstruction is made by
code from the local Info.cfg files; no memberships are distributed. Set
QC_VAL_FRACTION=0.20 only for the explicitly labelled 80/20 reconstruction.
The original scripts seed the stratified split, not every PyTorch training RNG.
No bitwise-identical retraining claim is made. The distributed final weights
are identified by SHA256SUMS and are the evaluated deployment checkpoints.

INFERENCE
python training/infer.py --help
Provide the local native phase image, optional zero-based 4D frame, final
checkpoint, architecture, patient ID and phase. Patient IDs exist only as
runtime inputs and output records. CPU is the default. Request CUDA explicitly
only on a machine where it is permitted. The input is resampled to 1.25 mm
in-plane, centre cropped/padded to 256 square, and slice-normalised exactly as
the original dataset code. Native predicted masks use geometry.py's original
hard-label inverse with nearest-neighbour interpolation and zero fill.

--mc uses twenty passes and the recorded per-phase CRC32 seed recipe. The
default slice-major order matches the submitted entropy procedure. The
revision B1 run used --loop phase-batched and a phase-sized batch when memory
allowed. To reproduce that run, set --batch-slices 0 for a whole phase, or
use the recorded batch size for the relevant architecture. These orders must
not be interchanged when comparing individual MC draws. A short pause is
applied for responsiveness. Per-pass native RV/LV volumes and processed-mask
hashes are saved; ED/ES pass ordinals are paired by analysis.mc_spread.

When reproducing the revision gate, --stored-mask supplies the saved processed
deterministic mask and --stored-entropy supplies its saved stride-two entropy
array. The gate computes entropy on the stored support (tolerance 1e-5), logs
argmax flips, and stops above a 0.1% phase voxel difference. New MC entropy is
aggregated on that same stored support. MC agreement with old draws is
descriptive. --save-masks saves masks only in the user's runtime output.

FINAL-CHECKPOINT DICE (TABLE S11)
python -m analysis.final_checkpoint_dice --development-root data/ACDC/database/training --weights weights --output outputs/final_checkpoint_dice.csv
This evaluates final weights on CPU, generates the reconstructed split locally,
and reports phase-mean 3D patient Dice per class. It does not train or select
checkpoints. --include-test-root can add the supplied held-out test partition.
