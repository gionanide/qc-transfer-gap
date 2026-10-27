C2 ORIGINAL-LABEL NNUNET

Conversion reads only the official ACDC development partition, with its original
labels: background 0, RV 1, myocardium 2, LV 3. ED/ES images become separate
nnU-Net cases. Image/label file contents are copied unchanged. The public source
contains no dataset files or memberships.

generate_splits.py reproduces the frozen five patient-disjoint diagnosis-
stratified folds with StratifiedKFold(shuffle=True, random_state=20260724).
Membership is generated locally from sorted development folders and Info.cfg.
ED and ES always belong to the same fold. No split list is distributed.

plans.json is the frozen C2 plan: official 2D configuration, batch size 56,
patch size 256 by 224, 1.5625 mm in-plane spacing. C2 used nnunetv2 2.8.1,
1000 epochs, 250 training and 50 validation iterations per epoch, with the
official optimization core. Fold initialization uses seed 20260804 + fold.
The qualification wrapper only added audit instrumentation; the portable
QualificationNNUNetTrainer preserves its name and official optimization core,
without the original machine-specific audit-output hooks. Retraining is not
guaranteed to regenerate identical bytes. Use the nine Release file identities
in SHA256SUMS for the actual evaluated weights.

training_inference.sh records preparation, fold training and inference commands.
The inference-only block is commented so executing the training recipe cannot
silently mix freshly trained and released weights. Training used eight data-
augmentation workers in the completed recovery run; these commands record that
setting. They are instructions, not processes started by this package.

Each fold is predicted with checkpoint_final.pth and --save_probabilities.
The official nnU-Net export resamples probabilities/logits to native geometry
before converting to labels and restoring crop/transpose geometry. C2 then
uses the official ensemble of the five separately exported native probability
arrays, with float32 averaging and final argmax. It does not use the fixed-
model hard-label nearest-neighbour inverse. Native volumes use those native
ensemble masks. Fold-spread scores use each fold's native predicted volumes.

processed_scores.py maps the native hard ensemble mask to the paper's grid
with nearest-neighbour interpolation for Dice. It maps each native probability
channel linearly, renormalises after crop/pad (empty padding becomes background),
and evaluates softmax entropy on stride-two pixels with the same aggregation
as the fixed models. For raw M&Ms references use --reference-labels mnms.
