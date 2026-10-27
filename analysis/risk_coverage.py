"""Deterministic risk-coverage and normalized deferral gain."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _finite_pair(scores: ArrayLike, errors: ArrayLike):
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    error = np.asarray(errors, dtype=np.float64).reshape(-1)
    if score.shape != error.shape:
        raise ValueError("scores and errors must have the same shape")
    valid = np.isfinite(score) & np.isfinite(error)
    return score[valid], error[valid]


def retained_risk_curve(scores: ArrayLike, errors: ArrayLike):
    """Defer highest scores first and return every nonempty coverage point."""
    score, error = _finite_pair(scores, errors)
    if score.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(-score, kind="mergesort")
    ordered_error = error[order]
    n = ordered_error.size
    retained_sum = np.cumsum(ordered_error[::-1])[::-1]
    retained_count = np.arange(n, 0, -1, dtype=np.float64)
    coverage = retained_count / n
    risk = retained_sum / retained_count
    return coverage, risk


def oracle_risk_curve(errors: ArrayLike):
    """Return the curve obtained by deferring the largest true errors first."""
    error = np.asarray(errors, dtype=np.float64).reshape(-1)
    error = error[np.isfinite(error)]
    return retained_risk_curve(error, error)


def random_risk_curve(errors: ArrayLike):
    """Return the expected random-deferral curve at every achievable coverage."""
    error = np.asarray(errors, dtype=np.float64).reshape(-1)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    coverage = np.arange(error.size, 0, -1, dtype=np.float64) / error.size
    return coverage, np.full(error.size, float(np.mean(error)), dtype=np.float64)


def discrete_aurc(risk: ArrayLike) -> float:
    """Average retained risk over all achievable nonempty coverage levels."""
    values = np.asarray(risk, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def normalized_deferral_gain(
    detector_aurc: float, random_aurc: float, oracle_aurc: float
) -> tuple[float, str]:
    denominator = float(random_aurc) - float(oracle_aurc)
    if not np.isfinite([detector_aurc, random_aurc, oracle_aurc]).all():
        return float("nan"), "nonfinite_input"
    if abs(denominator) <= np.finfo(np.float64).eps:
        return float("nan"), "zero_denominator"
    return (float(random_aurc) - float(detector_aurc)) / denominator, "defined"


def evaluate_deferral(scores: ArrayLike, errors: ArrayLike) -> dict[str, object]:
    score, error = _finite_pair(scores, errors)
    coverage, detector = retained_risk_curve(score, error)
    _, random = random_risk_curve(error)
    _, oracle = oracle_risk_curve(error)
    detector_aurc = discrete_aurc(detector)
    random_aurc = discrete_aurc(random)
    oracle_aurc = discrete_aurc(oracle)
    gain, status = normalized_deferral_gain(detector_aurc, random_aurc, oracle_aurc)
    return {
        "n": int(error.size),
        "coverage": coverage,
        "detector_risk": detector,
        "random_risk": random,
        "oracle_risk": oracle,
        "detector_aurc": detector_aurc,
        "random_aurc": random_aurc,
        "oracle_aurc": oracle_aurc,
        "normalized_deferral_gain": gain,
        "gain_status": status,
    }
