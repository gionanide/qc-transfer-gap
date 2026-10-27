"""Portable paths; preserved training constants. S11 uses a separate reconstruction."""
import os
from pathlib import Path
DATA_ROOT=Path(os.environ.get('QC_DATA_ROOT','data/ACDC/database'))
TRAIN_DIR=DATA_ROOT/'training'
TEST_DIR=DATA_ROOT/'testing'
OUT_DIR=Path(os.environ.get('QC_TRAIN_OUTPUT','outputs/training'))
NUM_CLASSES=4
LBL_RV,LBL_MYO,LBL_LV=1,2,3
TARGET_SPACING=(1.25,1.25)
PATCH_SIZE=(256,256)
VAL_FRACTION=float(os.environ.get('QC_VAL_FRACTION','0.15'))
SEED=42
BATCH_SIZE=16
MAX_EPOCHS=100
LR=1e-3
NUM_WORKERS=int(os.environ.get('QC_WORKERS','4'))
MYO_DENSITY=1.05
