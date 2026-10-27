"""
Per-patient clinical measurement evaluation.

This is the bridge from segmentation to clinical endpoint:
  predicted slices -> re-stack into a volume -> compute EDV/ESV/EF/mass
  -> compare to the SAME measures from ground-truth masks.

EF is a per-PATIENT, per-VOLUME quantity. We sum LV-cavity voxels across all
slices of the ED frame (EDV) and the ES frame (ESV). The error between
predicted-mask EF and ground-truth-mask EF is the target variable of the paper.
"""
import os
import numpy as np
import nibabel as nib
import torch

import config as C
from dataset import parse_info_cfg, frame_file, _resample_inplane, _crop_or_pad, _normalize


def _voxel_volume_ml(nii):
    sx, sy, sz = nii.header.get_zooms()[:3]
    return (sx * sy * sz) / 1000.0


def measures_from_mask_volume(mask_vol, vox_ml_ed, mask_vol_es, vox_ml_es):
    """Clinical measures from ED + ES label volumes (numpy int arrays)."""
    edv = (mask_vol == C.LBL_LV).sum() * vox_ml_ed
    esv = (mask_vol_es == C.LBL_LV).sum() * vox_ml_es
    ef = 100.0 * (edv - esv) / edv if edv > 0 else np.nan
    myo_mass = (mask_vol == C.LBL_MYO).sum() * vox_ml_ed * C.MYO_DENSITY
    return {"EDV": edv, "ESV": esv, "EF": ef, "MyoMass": myo_mass}


@torch.no_grad()
def predict_patient_volume(net, patient_dir, device):
    """Run the net slice-by-slice on a patient's ED and ES frames.
    Returns predicted label volumes (resampled grid) + voxel volumes,
    and the ground-truth measures (computed on the ORIGINAL grid)."""
    pid = os.path.basename(patient_dir)
    info = parse_info_cfg(patient_dir)
    net.eval()

    preds, gts, vox = {}, {}, {}
    for phase, fidx in (("ED", info["ED"]), ("ES", info["ES"])):
        img_nii = nib.load(frame_file(patient_dir, pid, fidx, gt=False))
        gt_nii = nib.load(frame_file(patient_dir, pid, fidx, gt=True))
        sx, sy = img_nii.header.get_zooms()[:2]
        vox_ml = _voxel_volume_ml(img_nii)

        img = np.asarray(img_nii.dataobj).astype(np.float32)   # H,W,S (original)
        gt = np.asarray(gt_nii.dataobj).astype(np.int64)
        n_slices = img.shape[2]

        pred_slices = []
        for s in range(n_slices):
            sl = _resample_inplane(img[:, :, s], (sx, sy), C.TARGET_SPACING, is_mask=False)
            sl = _crop_or_pad(sl, C.PATCH_SIZE, 0)
            sl = _normalize(sl)
            t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)  # 1,1,H,W
            logits = net(t)
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int64)
            pred_slices.append(pred)
        pred_vol = np.stack(pred_slices, axis=-1)  # H',W',S on resampled grid

        # GT measures on ORIGINAL grid (true clinical reference)
        preds[phase] = pred_vol
        gts[phase] = gt
        vox[phase] = vox_ml
        # voxel volume on the resampled grid (for predicted measures)
        vox[phase + "_pred"] = (C.TARGET_SPACING[0] * C.TARGET_SPACING[1] *
                                img_nii.header.get_zooms()[2]) / 1000.0

    pred_measures = measures_from_mask_volume(
        preds["ED"], vox["ED_pred"], preds["ES"], vox["ES_pred"])
    gt_measures = measures_from_mask_volume(
        gts["ED"], vox["ED"], gts["ES"], vox["ES"])
    return pred_measures, gt_measures, info["Group"]


@torch.no_grad()
def evaluate_ef(net, patient_dirs, device):
    """Loop patients -> table of predicted vs GT measures + errors."""
    rows = []
    for pdir in patient_dirs:
        pm, gm, group = predict_patient_volume(net, pdir, device)
        rows.append({
            "patient": os.path.basename(pdir), "group": group,
            "EF_gt": gm["EF"], "EF_pred": pm["EF"],
            "EF_abs_err": abs(pm["EF"] - gm["EF"]),
            "EDV_gt": gm["EDV"], "EDV_pred": pm["EDV"],
            "ESV_gt": gm["ESV"], "ESV_pred": pm["ESV"],
        })
    return rows


def summarize_ef(rows):
    errs = np.array([r["EF_abs_err"] for r in rows if not np.isnan(r["EF_abs_err"])])
    return {"EF_MAE": float(errs.mean()), "EF_MAE_std": float(errs.std()),
            "n": len(errs)}
