"""Exact and first-order propagation for EDV, SV, and EF.

EF is represented as a fraction only inside algebraic derivations. Public EF
values and EF errors are returned in percentage points.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float(value: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _scalar_if_scalar(value: NDArray[np.float64], *inputs: ArrayLike):
    if all(np.ndim(item) == 0 for item in inputs):
        return float(value)
    return value


def ef_points(edv_ml: ArrayLike, esv_ml: ArrayLike):
    """Return EF in percentage points, with invalid EDV reported as NaN."""
    edv = _as_float(edv_ml)
    esv = _as_float(esv_ml)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 100.0 * (edv - esv) / edv
    result = np.where((edv > 0.0) & np.isfinite(edv) & np.isfinite(esv), result, np.nan)
    return _scalar_if_scalar(result, edv_ml, esv_ml)


def exact_ef_error_points(
    reference_edv_ml: ArrayLike,
    reference_esv_ml: ArrayLike,
    delta_edv_ml: ArrayLike,
    delta_esv_ml: ArrayLike,
):
    """Return the exact EF error in points without clipping.

    Any observation with a nonpositive reference or predicted EDV is invalid
    and is represented by NaN.
    """
    d = _as_float(reference_edv_ml)
    s = _as_float(reference_esv_ml)
    delta_d = _as_float(delta_edv_ml)
    delta_s = _as_float(delta_esv_ml)
    predicted_d = d + delta_d
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 100.0 * (s * delta_d - d * delta_s) / (d * predicted_d)
    valid = (
        (d > 0.0)
        & (predicted_d > 0.0)
        & np.isfinite(d)
        & np.isfinite(s)
        & np.isfinite(delta_d)
        & np.isfinite(delta_s)
    )
    result = np.where(valid, result, np.nan)
    return _scalar_if_scalar(
        result, reference_edv_ml, reference_esv_ml, delta_edv_ml, delta_esv_ml
    )


def first_order_ef_error_points(
    reference_edv_ml: ArrayLike,
    reference_esv_ml: ArrayLike,
    delta_edv_ml: ArrayLike,
    delta_esv_ml: ArrayLike,
):
    """Return the first-order Taylor approximation to EF error in points."""
    d = _as_float(reference_edv_ml)
    s = _as_float(reference_esv_ml)
    delta_d = _as_float(delta_edv_ml)
    delta_s = _as_float(delta_esv_ml)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 100.0 * ((s / np.square(d)) * delta_d - delta_s / d)
    valid = (
        (d > 0.0)
        & np.isfinite(d)
        & np.isfinite(s)
        & np.isfinite(delta_d)
        & np.isfinite(delta_s)
    )
    result = np.where(valid, result, np.nan)
    return _scalar_if_scalar(
        result, reference_edv_ml, reference_esv_ml, delta_edv_ml, delta_esv_ml
    )


def exact_from_first_order_points(
    first_order_error_points: ArrayLike,
    reference_edv_ml: ArrayLike,
    delta_edv_ml: ArrayLike,
):
    """Apply the exact multiplicative correction to a first-order EF error."""
    first = _as_float(first_order_error_points)
    d = _as_float(reference_edv_ml)
    delta_d = _as_float(delta_edv_ml)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = first / (1.0 + delta_d / d)
    valid = (
        (d > 0.0)
        & (d + delta_d > 0.0)
        & np.isfinite(first)
        & np.isfinite(d)
        & np.isfinite(delta_d)
    )
    result = np.where(valid, result, np.nan)
    return _scalar_if_scalar(
        result, first_order_error_points, reference_edv_ml, delta_edv_ml
    )


def approximation_residual_points(
    exact_error_points: ArrayLike, first_order_error_points: ArrayLike
):
    """Return exact minus first-order EF error in percentage points."""
    exact = _as_float(exact_error_points)
    first = _as_float(first_order_error_points)
    result = exact - first
    return _scalar_if_scalar(result, exact_error_points, first_order_error_points)


def edv_error_ml(delta_edv_ml: ArrayLike):
    """Identity propagation for EDV error."""
    result = _as_float(delta_edv_ml)
    return _scalar_if_scalar(result, delta_edv_ml)


def sv_error_ml(delta_edv_ml: ArrayLike, delta_esv_ml: ArrayLike):
    """Exact propagation for SV error."""
    result = _as_float(delta_edv_ml) - _as_float(delta_esv_ml)
    return _scalar_if_scalar(result, delta_edv_ml, delta_esv_ml)
