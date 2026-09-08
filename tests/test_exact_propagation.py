from __future__ import annotations

import numpy as np

from cardiac_functional_qc.propagation import (
    ef_points,
    exact_ef_error_points,
    exact_from_first_order_points,
    first_order_ef_error_points,
)


def test_golden_case_a() -> None:
    exact = exact_ef_error_points(100.0, 40.0, 10.0, 0.0)
    first = first_order_ef_error_points(100.0, 40.0, 10.0, 0.0)
    assert ef_points(100.0, 40.0) == 60.0
    assert np.isclose(ef_points(110.0, 40.0), 63.63636363636363)
    assert np.isclose(exact, 3.6363636363636362)
    assert first == 4.0
    assert np.isclose(exact_from_first_order_points(first, 100.0, 10.0), exact)


def test_golden_case_b() -> None:
    exact = exact_ef_error_points(100.0, 40.0, 10.0, 10.0)
    first = first_order_ef_error_points(100.0, 40.0, 10.0, 10.0)
    assert np.isclose(exact, -5.454545454545454)
    assert np.isclose(first, -6.0, atol=1e-12, rtol=0.0)


def test_exact_formula_matches_direct_difference() -> None:
    rng = np.random.default_rng(21)
    d = rng.uniform(60.0, 220.0, 100)
    s = rng.uniform(10.0, 0.8 * d)
    delta_d = rng.normal(0.0, 8.0, 100)
    delta_s = rng.normal(0.0, 8.0, 100)
    direct = ef_points(d + delta_d, s + delta_s) - ef_points(d, s)
    exact = exact_ef_error_points(d, s, delta_d, delta_s)
    assert np.allclose(direct, exact, atol=1e-12, rtol=0.0)
