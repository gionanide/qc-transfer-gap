from __future__ import annotations

import numpy as np

from cardiac_functional_qc.propagation import ef_points, exact_ef_error_points


def test_nonpositive_predicted_edv_is_invalid() -> None:
    assert np.isnan(exact_ef_error_points(100.0, 40.0, -100.0, 0.0))
    assert np.isnan(exact_ef_error_points(100.0, 40.0, -101.0, 0.0))


def test_nonpositive_reference_edv_is_invalid() -> None:
    assert np.isnan(ef_points(0.0, 0.0))
    assert np.isnan(exact_ef_error_points(0.0, 0.0, 1.0, 0.0))
