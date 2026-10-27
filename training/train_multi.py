"""
train_multi.py -- train any architecture in the zoo on ACDC, save best-by-EF checkpoint.
Mirrors train.py exactly; only the network and checkpoint name change.

    python train_multi.py --arch segresnet
    python train_multi.py --arch attnunet
    python train_multi.py --arch unet              # reproduces the baseline
    python train_multi.py --arch swinunetr --amp   # transformer, mixed precision

--amp enables TF32 + autocast mixed precision (a real ~1.5-2x on the transformer on
Ampere+). With --amp OFF the loop is bit-identical to the original FP32 training, so the
convolutional baselines remain exactly reproducible; only the transformer run uses it.

Saves runs/best_by_ef_{arch}.pt  (and runs/last_{arch}.pt).
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

import config as C
from dataset import stratified_split, build_slice_index, ACDCSliceDataset
from model_zoo import build_model, ARCHS
from eval_ef import evaluate_ef, summarize_ef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--amp", action="store_true",
                    help="enable TF32 + mixed precision (used for the transformer run)")
    args = ap.parse_args()
    arch = args.arch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  arch: {arch}")

    # cuDNN kernel autotuning: fixed 256x256 input -> picks fast kernels.
    # Not a precision change, so it is safe to leave on for every arch.
    torch.backends.cudnn.benchmark = True

    # Precision optimizations, gated so they apply ONLY to the --amp (transformer) run.
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[amp] TF32 + autocast mixed precision ENABLED for this run")

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pat, val_pat = stratified_split(C.TRAIN_DIR, C.VAL_FRACTION, C.SEED)
    print(f"train patients: {len(train_pat)}  val patients: {len(val_pat)}")
    train_recs = build_slice_index(train_pat)
    train_loader = DataLoader(ACDCSliceDataset(train_recs, augment=True),
                              batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True)

    net = build_model(arch).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(net.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.MAX_EPOCHS)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # torch>=2.3 API. For older torch use: torch.cuda.amp.GradScaler(enabled=use_amp)
    # and torch.cuda.amp.autocast(enabled=use_amp).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_ef_mae = float("inf")
    for epoch in range(1, C.MAX_EPOCHS + 1):
        net.train(); running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = batch["label"].unsqueeze(1).to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_fn(net(img), lbl)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()
        train_loss = running / len(train_loader)

        net.eval(); dice_metric.reset()
        with torch.no_grad():
            val_recs = build_slice_index(val_pat)
            val_loader = DataLoader(ACDCSliceDataset(val_recs, augment=False),
                                    batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS)
            for batch in val_loader:
                img = batch["image"].to(device)
                lbl = batch["label"].unsqueeze(1).to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    logits = net(img)
                pred = logits.argmax(1, keepdim=True)
                dice_metric(one_hot(pred, C.NUM_CLASSES), one_hot(lbl, C.NUM_CLASSES))
            val_dice = dice_metric.aggregate().item()

        ef_str = ""
        if epoch % 5 == 0 or epoch == C.MAX_EPOCHS:
            # evaluate_ef runs in its own precision; EF is a measurement, not a trained
            # quantity, so this does not affect checkpoint selection.
            ef = summarize_ef(evaluate_ef(net, val_pat, device))
            ef_str = f" | val EF-MAE {ef['EF_MAE']:.2f}±{ef['EF_MAE_std']:.2f}"
            if ef["EF_MAE"] < best_ef_mae:
                best_ef_mae = ef["EF_MAE"]
                torch.save(net.state_dict(), C.OUT_DIR / f"best_by_ef_{arch}.pt")
        print(f"[{arch}] epoch {epoch:3d} | loss {train_loss:.4f} | "
              f"val Dice {val_dice:.4f}{ef_str}")

    torch.save(net.state_dict(), C.OUT_DIR / f"last_{arch}.pt")
    print(f"[{arch}] done. best val EF-MAE {best_ef_mae:.2f} "
          f"-> best_by_ef_{arch}.pt")


if __name__ == "__main__":
    main()
