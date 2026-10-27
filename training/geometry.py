"""Exact study grid functions: linear image / nearest-label resampling and centred crop/pad."""
from __future__ import annotations
from typing import List,Tuple,Sequence
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import zoom
C=SimpleNamespace(TARGET_SPACING=(1.25,1.25),PATCH_SIZE=(256,256))
def _resample_inplane(slice2d, src_spacing, target_spacing, is_mask):
    """Simple in-plane resample by zoom. Nearest for masks, linear for images."""
    from scipy.ndimage import zoom
    zoom_factors = (src_spacing[0] / target_spacing[0],
                    src_spacing[1] / target_spacing[1])
    order = 0 if is_mask else 1
    return zoom(slice2d, zoom_factors, order=order)

def _crop_or_pad(arr, size, pad_value=0):
    """Center crop or pad a 2D array to `size`."""
    out = np.full(size, pad_value, dtype=arr.dtype)
    h, w = arr.shape
    th, tw = size
    # crop
    sh = max(0, (h - th) // 2); sw = max(0, (w - tw) // 2)
    arr_c = arr[sh:sh + th, sw:sw + tw]
    # pad into center
    ch, cw = arr_c.shape
    dh = (th - ch) // 2; dw = (tw - cw) // 2
    out[dh:dh + ch, dw:dw + cw] = arr_c
    return out

def _normalize(img2d):
    """Z-score per slice on nonzero region; robust to empty slices."""
    m = img2d[img2d > 0]
    if m.size == 0:
        return img2d.astype(np.float32)
    mean, std = m.mean(), m.std() + 1e-6
    return ((img2d - mean) / std).astype(np.float32)
def preprocess_slice(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    image_resampled = _resample_inplane(image_slice, (sx, sy), C.TARGET_SPACING, is_mask=False)
    label_resampled = _resample_inplane(label_slice, (sx, sy), C.TARGET_SPACING, is_mask=True)
    resampled_shape = tuple(int(x) for x in image_resampled.shape)
    image = _normalize(_crop_or_pad(image_resampled, C.PATCH_SIZE, pad_value=0))
    label = _crop_or_pad(label_resampled, C.PATCH_SIZE, pad_value=0).astype(np.int16)
    return image, label, resampled_shape

def preprocess_volume(
    image_native: np.ndarray,
    label_native: np.ndarray,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    images = []
    labels = []
    shapes: List[Tuple[int, int]] = []
    for z in range(image_native.shape[2]):
        image, label, shape = preprocess_slice(image_native[:, :, z], label_native[:, :, z], sx, sy)
        images.append(image)
        labels.append(label)
        shapes.append(shape)
    return np.stack(images, axis=0), np.stack(labels, axis=0), shapes

def invert_crop_or_pad(patch: np.ndarray, resampled_shape: Tuple[int, int]) -> np.ndarray:
    h, w = resampled_shape
    th, tw = C.PATCH_SIZE
    sh = max(0, (h - th) // 2)
    sw = max(0, (w - tw) // 2)
    ch = min(h, th)
    cw = min(w, tw)
    dh = (th - ch) // 2
    dw = (tw - cw) // 2
    out = np.zeros((h, w), dtype=patch.dtype)
    out[sh : sh + ch, sw : sw + cw] = patch[dh : dh + ch, dw : dw + cw]
    return out

def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.int16, copy=False)
    zoom_factors = (shape[0] / max(mask.shape[0], 1), shape[1] / max(mask.shape[1], 1))
    resized = zoom(mask, zoom_factors, order=0)
    return _crop_or_pad(resized, shape, pad_value=0).astype(np.int16)

def processed_prediction_to_native(
    pred_processed: np.ndarray,
    resampled_shapes: Sequence[Tuple[int, int]],
    native_shape: Tuple[int, int, int],
) -> np.ndarray:
    native_slices = []
    for z in range(pred_processed.shape[0]):
        uncropped = invert_crop_or_pad(pred_processed[z], resampled_shapes[z])
        native_slices.append(resize_mask_to_shape(uncropped, native_shape[:2]))
    return np.stack(native_slices, axis=2).astype(np.int16)

def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool, copy=False)
    bb = b.astype(bool, copy=False)
    den = int(aa.sum()) + int(bb.sum())
    if den == 0:
        return float("nan")
    return float(2.0 * np.logical_and(aa, bb).sum() / den)

def volume_ml(mask: np.ndarray, voxel_ml: float) -> float:
    return float(np.count_nonzero(mask) * voxel_ml)

def ef(edv: float, esv: float) -> float:
    if not np.isfinite(edv) or edv <= 0:
        return float("nan")
    return float(100.0 * (edv - esv) / edv)
