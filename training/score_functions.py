"""Preserved uncertainty weights and aggregation on the stride-two score grid."""
import numpy as np
from scipy.ndimage import binary_dilation
def struct_weight(pred_stack, lbl, meas_aware):
    """naive = uniform over structure; meas_aware = dilated structure x slice area-share."""
    reg = (pred_stack == lbl)
    if not meas_aware:
        return reg.astype(float)
    regd = np.stack([binary_dilation(reg[s], iterations=2) for s in range(reg.shape[0])], 0)
    area = reg.reshape(reg.shape[0], -1).sum(1).astype(float)
    share = area / (area.sum() + 1e-8)
    return regd.astype(float) * share[:, None, None]

def agg(unc, w):
    if w.sum() > 0:
        return float((unc * w).sum() / w.sum())
    m = w > 0
    return float(unc[m].mean()) if m.any() else float(unc.mean())
def enable_dropout(net):
    import torch
    """Set ONLY dropout layers to train mode (keep batchnorm in eval)."""
    for m in net.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
