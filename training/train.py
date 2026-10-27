"""
Train the 2D U-Net on ACDC and validate with BOTH:
  * per-slice Dice (segmentation quality)
  * per-patient EF MAE (clinical-measurement quality)  <- the paper's bridge

Run:
    python train.py
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

import config as C
from dataset import (stratified_split, build_slice_index, ACDCSliceDataset,
                     list_patients)
from model import build_unet
from eval_ef import evaluate_ef, summarize_ef


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_pat, val_pat = stratified_split(C.TRAIN_DIR, C.VAL_FRACTION, C.SEED)
    print(f"train patients: {len(train_pat)}  val patients: {len(val_pat)}")
    train_recs = build_slice_index(train_pat)
    print(f"train slices: {len(train_recs)}")

    train_ds = ACDCSliceDataset(train_recs, augment=True)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True)

    # --- model / loss / optim ---
    net = build_unet().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(net.parameters(), lr=C.LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_ef_mae = float("inf")
    for epoch in range(1, C.MAX_EPOCHS + 1):
        net.train()
        running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = batch["label"].unsqueeze(1).to(device)  # B,1,H,W
            opt.zero_grad()
            logits = net(img)
            loss = loss_fn(logits, lbl)
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        # --- per-slice Dice on val ---
        net.eval()
        dice_metric.reset()
        with torch.no_grad():
            val_recs = build_slice_index(val_pat)
            val_loader = DataLoader(ACDCSliceDataset(val_recs, augment=False),
                                    batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS)
            for batch in val_loader:
                img = batch["image"].to(device)
                lbl = batch["label"].unsqueeze(1).to(device)
                pred = net(img).argmax(1, keepdim=True)
                dice_metric(one_hot(pred, C.NUM_CLASSES), one_hot(lbl, C.NUM_CLASSES))
            val_dice = dice_metric.aggregate().item()

        # --- per-patient EF MAE on val (every 5 epochs to save time) ---
        ef_str = ""
        if epoch % 5 == 0 or epoch == C.MAX_EPOCHS:
            rows = evaluate_ef(net, val_pat, device)
            ef = summarize_ef(rows)
            ef_str = f" | val EF-MAE {ef['EF_MAE']:.2f}±{ef['EF_MAE_std']:.2f}"
            if ef["EF_MAE"] < best_ef_mae:
                best_ef_mae = ef["EF_MAE"]
                torch.save(net.state_dict(), C.OUT_DIR / "best_by_ef.pt")

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val Dice {val_dice:.4f}{ef_str}")

    torch.save(net.state_dict(), C.OUT_DIR / "last.pt")
    print(f"done. best val EF-MAE: {best_ef_mae:.2f}  (checkpoints in {C.OUT_DIR})")


if __name__ == "__main__":
    main()
