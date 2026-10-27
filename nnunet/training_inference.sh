#!/usr/bin/env bash
# nnU-Net baseline: original labels, fixed plans, locally generated folds; run from repo root.
set -euo pipefail
: "${ACDC_DEVELOPMENT_ROOT:?Set the local original-label development folder}"
export nnUNet_raw="${nnUNet_raw:-work/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-work/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-work/nnUNet_results}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"

# Preparation for training; never use refined labels.
python -m nnunet.convert_dataset --development-root "$ACDC_DEVELOPMENT_ROOT" --output "$nnUNet_raw/Dataset701_ACDCDevelopment"
nnUNetv2_plan_and_preprocess -d 701 -c 2d --verify_dataset_integrity -np 1
cp nnunet/plans.json "$nnUNet_preprocessed/Dataset701_ACDCDevelopment/nnUNetPlans.json"
python -m nnunet.generate_splits --development-root "$ACDC_DEVELOPMENT_ROOT" --output "$nnUNet_preprocessed/Dataset701_ACDCDevelopment/splits_final.json"
python -m nnunet.register_trainer --install
for fold in 0 1 2 3 4; do
  python -m nnunet.train_fold --fold "$fold" --device cpu
done

# Inference only with the downloaded study checkpoints: do not run the block
# above if you only want inference. Copy plans/dataset.json next to the folds.
# MODEL_ROOT="$nnUNet_results/Dataset701_ACDCDevelopment/QualificationNNUNetTrainer__nnUNetPlans__2d"
# mkdir -p "$MODEL_ROOT"
# cp nnunet/plans.json "$MODEL_ROOT/plans.json"
# cp nnunet/dataset.json "$MODEL_ROOT/dataset.json"
# for fold in 0 1 2 3 4; do
#   mkdir -p "$MODEL_ROOT/fold_$fold"
#   cp "weights/nnunet_fold_${fold}_checkpoint_final.pth" "$MODEL_ROOT/fold_$fold/checkpoint_final.pth"
#   nnUNetv2_predict -i "$INFERENCE_IMAGES" -o "outputs/nnunet/fold_$fold" -d 701 -c 2d -tr QualificationNNUNetTrainer -f "$fold" -chk checkpoint_final.pth --save_probabilities -npp 1 -nps 1 -device cpu
# done
# Average the separately exported native probabilities with the official
# ensemble, as in the study. This preserves each fold's native export geometry.
# nnUNetv2_ensemble -i outputs/nnunet/fold_0 outputs/nnunet/fold_1 outputs/nnunet/fold_2 outputs/nnunet/fold_3 outputs/nnunet/fold_4 -o outputs/nnunet/ensemble -np 1 --save_npz
