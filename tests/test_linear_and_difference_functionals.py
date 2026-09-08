from __future__ import annotations

import numpy as np

from cardiac_functional_qc.propagation import edv_error_ml, sv_error_ml


def test_edv_is_identity_and_sv_is_phase_difference() -> None:
    delta_d = np.array([-5.0, 0.0, 8.0])
    delta_s = np.array([2.0, -3.0, 8.0])
    assert np.array_equal(edv_error_ml(delta_d), delta_d)
    assert np.array_equal(sv_error_ml(delta_d, delta_s), np.array([-7.0, 3.0, 0.0]))
