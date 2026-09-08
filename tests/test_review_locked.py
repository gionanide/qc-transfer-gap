"""Independent policy checks, including cases absent from the locked cohort."""

import itertools
import unittest

import numpy as np

from cardiac_functional_qc._locked_review import evaluate, flags_and_order, risk_of_retained, take_ranked


def explicit_policies(counts, errors, scores, ids, flags, flag_order, k):
    """Small expanded-list reference implementation, no multiplicity algebra."""
    expanded = np.repeat(np.arange(len(counts)), counts).tolist()
    f_rank = {int(p): i for i, p in enumerate(flag_order)}
    flagged = sorted([p for p in expanded if flags[p]], key=lambda p: f_rank[p])
    qc = sorted(expanded, key=lambda p: (-scores[p], ids[p]))
    oracle = sorted(expanded, key=lambda p: (-errors[p], ids[p]))
    reviewed_flags = flagged[:k]
    unflagged = [p for p in expanded if not flags[p]]
    pq = reviewed_flags + sorted(unflagged, key=lambda p: (-scores[p], ids[p]))[:k-len(reviewed_flags)]

    def risk(removed):
        rest = expanded.copy()
        for p in removed:
            rest.remove(p)
        return float(np.mean(errors[rest]))

    # Enumerate equally likely sets of occurrence positions, including duplicates.
    random = np.mean([np.mean(errors[[p for i, p in enumerate(expanded) if i not in selected]])
                      for selected in itertools.combinations(range(len(expanded)), k)])
    secondary = k - len(reviewed_flags)
    pr = np.mean([risk(reviewed_flags + [unflagged[i] for i in selected])
                  for selected in itertools.combinations(range(len(unflagged)), secondary)])
    return np.array([random, risk(qc[:k]), pr, risk(pq), risk(oracle[:k])])


class PoliciesTest(unittest.TestCase):
    def test_finite_flags_duplicates_ties_and_overflow_against_enumeration(self):
        ids = np.array(["b", "a", "d", "c", "f", "e", "g"])
        edv = np.full(7, 100.)
        esv = np.array([50., 150., 250., 120., 40., 40., 30.])
        ef = 100 * (1 - esv / edv)
        errors = np.array([5., 70., 210., 23., 12., 7., 10.])
        scores = np.array([.2, .8, .5, .5, .1, .1, .7])
        flags, order, _, _ = flags_and_order(edv, esv, ef, ids)
        rng = np.random.default_rng(914)
        counts = np.vstack([np.ones(7, int)] + [np.bincount(rng.integers(0, 7, 7), minlength=7) for _ in range(35)])
        for k in (0, 1, 2, 3, 5, 6):
            actual, nf, _ = evaluate(counts, errors, scores, ids, flags, order, k)
            for i, sample in enumerate(counts):
                expected = explicit_policies(sample, errors, scores, ids, flags, order, k)
                np.testing.assert_allclose(actual[i], expected, rtol=0, atol=1e-10)
            assert not nf.any()

    def test_nonfinite_priority_and_finite_severity(self):
        ids = np.array(["z", "a", "f", "d", "b", "c", "g"])
        edv = [0., 20., 10., 10., 10., 10., 10.]
        esv = [1., np.nan, 30., 15., 30., -10., 5.]
        ef = [np.nan, np.nan, -200., -50., -200., 200., 50.]
        flags, order, _, _ = flags_and_order(edv, esv, ef, ids)
        self.assertEqual(order.tolist(), [1, 0, 4, 2, 5, 3])
        self.assertFalse(flags[-1])
        counts = np.ones((1, 7), int)
        self.assertEqual(np.flatnonzero(take_ranked(counts, order, 3)[0]).tolist(), [0, 1, 4])

    def test_zero_flags_exact_policies_and_floor_budget(self):
        ids = np.array(list("abcdefg"))
        errors = np.arange(7, dtype=float)
        flags, order, _, _ = flags_and_order(np.ones(7), np.zeros(7), np.full(7, 100.), ids)
        counts = np.array([[1]*7, [2, 0, 0, 3, 1, 1, 0]])
        for budget, expected_k in [(.1, 0), (.2, 1)]:
            k = int(np.floor(budget * 7))
            self.assertEqual(k, expected_k)
            r, _, _ = evaluate(counts, errors, np.ones(7), ids, flags, order, k)
            np.testing.assert_allclose(r[:, 0], r[:, 2], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(r[:, 1], r[:, 3])

    def test_nonfinite_retention_is_undefined_with_expected_count(self):
        errors = np.array([np.nan, 2., 4.])
        weights = np.array([[0., 1., 1.], [1., 0., 1.], [2/3]*3])
        r, nf = risk_of_retained(weights, errors)
        self.assertEqual(r[0], 3.)
        self.assertTrue(np.isnan(r[1:]).all())
        np.testing.assert_allclose(nf, [0., 1., 2/3])

    def test_finite_extremes_are_never_clipped(self):
        errors = np.array([0., 1., 10000.])
        r, nf = risk_of_retained(np.ones((1, 3)), errors)
        self.assertEqual(r[0], 10001/3)
        self.assertEqual(nf[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
