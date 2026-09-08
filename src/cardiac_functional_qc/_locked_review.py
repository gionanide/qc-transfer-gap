"""Exploratory matched-workload RV EF deferral analysis of locked paper outputs.

Run from any directory: python analyze.py
Only config-pinned inputs are read; all artifacts stay beside this script.
Bootstrap multiplicities preserve repeated patients as separate sample units.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
STRATEGIES = ["random", "qc", "plausibility_random", "plausibility_qc", "oracle"]
METRICS = STRATEGIES + ["delta1", "delta2"] + ["reduction_" + s for s in STRATEGIES]
FLAG_COLUMNS = ["predicted_edv_le_zero", "predicted_esv_gt_predicted_edv",
                "predicted_ef_outside_0_100", "nonfinite_predicted_ef"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flags_and_order(edv, esv, ef, ids):
    """Prediction-only overflow ordering, with patient-ID tie breaking."""
    edv, esv, ef = [np.asarray(x, float) for x in (edv, esv, ef)]
    individual = np.column_stack((edv <= 0, esv > edv,
                                 (ef < 0) | (ef > 100), ~np.isfinite(ef)))
    flags = individual.any(axis=1)
    high_priority = (~np.isfinite(edv) | ~np.isfinite(esv) |
                     ~np.isfinite(ef) | (edv <= 0))
    severity = np.zeros(len(edv))
    finite_ratio = np.isfinite(edv) & (edv > 0) & np.isfinite(esv)
    ratio = esv[finite_ratio] / edv[finite_ratio]
    severity[finite_ratio] = np.maximum.reduce((np.zeros(len(ratio)), -ratio, ratio - 1))
    # All high-priority cases tie on severity; patient ID decides their order.
    sorting_severity = np.where(high_priority, 0., severity)
    ordered = np.lexsort((np.asarray(ids, str), -sorting_severity, ~high_priority))
    return flags, ordered[flags[ordered]], individual, severity


def take_ranked(counts, order, slots):
    """Allocate integer slots in rank order, including partial duplicate counts."""
    slots = np.broadcast_to(np.asarray(slots, int), (len(counts),))
    selected = np.zeros_like(counts)
    if len(order):
        available = counts[:, order]
        before = np.cumsum(available, axis=1) - available
        selected[:, order] = np.minimum(available, np.maximum(0, slots[:, None] - before))
    return selected


def risk_of_retained(retained_weights, errors):
    """Also supports undefined expected risk; never silently nanmeans or caps."""
    finite = np.isfinite(errors)
    nf_count = retained_weights[:, ~finite].sum(axis=1)
    denominator = retained_weights.sum(axis=1)
    numerator = retained_weights[:, finite] @ errors[finite]
    risk = np.divide(numerator, denominator, out=np.full(len(denominator), np.nan),
                     where=(denominator > 0) & (nf_count == 0))
    return risk, nf_count


def evaluate(counts, errors, scores, ids, flags, flag_order, k):
    """Evaluate all policies on the identical matrix of bootstrap multiplicities."""
    n = counts.sum(axis=1)
    if np.any((k < 0) | (k >= n)):
        raise ValueError("Review count must leave at least one patient unreviewed")
    qc_order = np.lexsort((ids, -scores))
    # The oracle is non-deployable. Nonfinite error has highest priority if supplied.
    oracle_order = np.lexsort((ids, -np.where(np.isfinite(errors), errors, np.inf)))
    qc_reviewed = take_ranked(counts, qc_order, k)
    oracle_reviewed = take_ranked(counts, oracle_order, k)
    flag_reviewed = take_ranked(counts, flag_order, k)
    secondary_slots = k - flag_reviewed.sum(axis=1)
    pq_reviewed = flag_reviewed + take_ranked(counts, qc_order[~flags[qc_order]], secondary_slots)
    flag_n = counts[:, flags].sum(axis=1)
    unflag_n = n - flag_n
    # Flags not fitting are retained deterministically; all unflagged patients
    # have equal probability of filling the remaining slots when flags fit.
    fraction = np.divide(secondary_slots, unflag_n, out=np.zeros(len(n)), where=unflag_n > 0)
    pr_retained = (counts - flag_reviewed).astype(float)
    pr_retained[:, ~flags] *= (1 - fraction[:, None])
    retained = [counts * ((n - k) / n)[:, None], counts - qc_reviewed,
                pr_retained, counts - pq_reviewed, counts - oracle_reviewed]
    for weights in retained:
        np.testing.assert_allclose(weights.sum(axis=1), n - k, rtol=0, atol=1e-10)
        if np.any(weights < 0):
            raise AssertionError("Negative retained weight")
    for reviewed in (qc_reviewed, pq_reviewed, oracle_reviewed):
        np.testing.assert_array_equal(reviewed.sum(axis=1), np.full(len(counts), k))
        assert np.all(reviewed <= counts)
    result = [risk_of_retained(w, errors) for w in retained]
    risks = np.column_stack([r[0] for r in result])
    nonfinite = np.column_stack([r[1] for r in result])
    # For random policies these are exact expectations, not random simulations.
    point_review_probability = np.stack([1 - w[0] / counts[0] for w in retained], axis=1)
    return risks, nonfinite, point_review_probability


def derived_metrics(risks):
    return np.concatenate((risks, (risks[..., 2] - risks[..., 3])[..., None],
                           (risks[..., 1] - risks[..., 2])[..., None],
                           risks[..., 0, None] - risks), axis=-1)


def summarize(vector):
    # Do not silently discard undefined bootstrap replicates.
    finite = np.isfinite(vector[1:])
    if not finite.all():
        raise ValueError("Undefined bootstrap estimate; report instead of conditioning it away")
    low, high = np.quantile(vector[1:], [.025, .975], method="linear")
    return float(vector[0]), float(low), float(high)


def load_inputs(config):
    provenance = {}
    for key, record in config["inputs"].items():
        path = ROOT / record["path"]
        actual = sha(path)
        if actual != record["sha256"]:
            raise ValueError(f"Locked hash mismatch: {key}: {path}")
        provenance[key] = {**record, "absolute_path": str(path), "hash_verified": True}
    architectures, scores = config["architectures"], config["scores"]
    tables, audit, align = {}, [], []
    for cohort, count in config["cohorts"].items():
        prefix = cohort.lower()
        p = pd.read_csv(ROOT / config["inputs"][prefix + "_patient_table"]["path"])
        q = pd.read_csv(ROOT / config["inputs"][prefix + "_qc_table"]["path"])
        keys = ["patient_id", "architecture"]
        for table, name in [(p, "patient"), (q, "QC")]:
            assert not table[keys].isna().any().any()
            assert not table.duplicated(keys).any()
            assert set(table.architecture) == set(architectures)
            for arch in architectures:
                a = table.loc[table.architecture.eq(arch)]
                assert len(a) == count
                assert a.checkpoint_sha256.eq(config["checkpoint_sha256"][arch]).all()
            if name == "QC":
                assert q.mc_dropout_passes.eq(20).all()
                assert q.phase_aggregation.eq("arithmetic mean of ED and ES").all()
        assert set(map(tuple, p[keys].values)) == set(map(tuple, q[keys].values))
        p = p.sort_values(keys).reset_index(drop=True)
        q = q.sort_values(keys).reset_index(drop=True)
        for score in scores:
            if score in p:
                np.testing.assert_array_equal(p[score], q[score])
        m = p.drop(columns=[s for s in scores if s in p]).merge(q[keys + scores], on=keys, validate="one_to_one")
        m["cohort"] = cohort
        required = ["pred_edv_ml", "pred_esv_ml", "pred_sv_ml", "pred_rvef_points",
                    "ref_edv_ml", "ref_esv_ml", "ref_sv_ml", "ref_rvef_points",
                    "delta_rvef_points", "abs_delta_rvef_points"] + scores
        for arch in architectures:
            a = m.loc[m.architecture.eq(arch)]
            for col in required:
                x = a[col].to_numpy(float)
                audit.append(dict(cohort=cohort, architecture=arch, field=col, n=len(x),
                                  missing_or_nan=int(np.isnan(x).sum()), posinf=int(np.isposinf(x).sum()),
                                  neginf=int(np.isneginf(x).sum()), nonfinite_total=int((~np.isfinite(x)).sum())))
        # Existing score/error finite-pair eligibility is identical for all 16
        # combinations in each locked cohort. No cohort intersection is invented.
        eligible = np.isfinite(m[["abs_delta_rvef_points"] + scores].to_numpy()).all(axis=1)
        assert eligible.all(), "Pinned source unexpectedly needs eligibility exclusions"
        assert np.isfinite(m[required].to_numpy()).all()
        np.testing.assert_allclose(m.abs_delta_rvef_points, abs(m.pred_rvef_points - m.ref_rvef_points), rtol=0, atol=1e-10)
        for prefix in ["ref", "pred"]:
            np.testing.assert_allclose(m[prefix + "_rvef_points"],
                                       100 * (1 - m[prefix + "_esv_ml"] / m[prefix + "_edv_ml"]), rtol=0, atol=1e-10)
        id_sets = [set(m.loc[m.architecture.eq(a), "patient_id"]) for a in architectures]
        assert all(ids == id_sets[0] for ids in id_sets)
        if cohort == "ACDC":
            assert id_sets[0] == {f"patient{i}" for i in range(101, 151)}
        for col in ["ref_edv_ml", "ref_esv_ml", "ref_rvef_points"]:
            assert m.groupby("patient_id")[col].agg(lambda x: np.ptp(x.to_numpy())).max() < 1e-10
        align.append(dict(cohort=cohort, n_patients=count, n_patient_architecture_rows=len(m),
                          unique_keys=True, same_ids_all_architectures=True, score_alignment=True,
                          reference_alignment=True, excluded=0))
        tables[cohort] = m
    pd.DataFrame(audit).to_csv(OUT / "nonfinite_audit.csv", index=False)
    pd.DataFrame(align).to_csv(OUT / "patient_alignment.csv", index=False)
    write_json(OUT / "input_provenance.json", provenance)
    return tables


def run():
    started = datetime.now(timezone.utc)
    OUT.mkdir(exist_ok=True)
    config = json.loads((HERE / "config.json").read_text())
    tables = load_inputs(config)
    print("Input hashes, finite-value audit, and patient alignment verified.", flush=True)
    arches, scores, budgets = config["architectures"], config["scores"], config["budgets"]
    boot_n = config["bootstrap_replicates"]
    architecture_rows, main_rows, capacity, selections, checks, flagged_cases = [], [], [], [], [], []
    raw_arrays, sample_arrays = {}, {}
    expected_flags = pd.read_csv(ROOT / config["inputs"]["locked_flag_cases"]["path"])
    library_path = ROOT / config["inputs"]["risk_coverage_source"]["path"]
    spec = importlib.util.spec_from_file_location("locked_risk_coverage", library_path)
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    for cohort, table in tables.items():
        n = config["cohorts"][cohort]
        ids = np.array(sorted(table.patient_id.unique()))
        rng = np.random.default_rng(config["cohort_seeds"][cohort])
        sampled = rng.integers(0, n, size=(boot_n, n), dtype=np.int32)
        sample_arrays[cohort + "_patient_ids"] = ids
        sample_arrays[cohort + "_indices"] = sampled
        counts = np.zeros((boot_n + 1, n), dtype=np.int32)
        counts[0] = 1
        np.add.at(counts[1:], (np.arange(boot_n)[:, None], sampled), 1)
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(boot_n + 1, n))
        risks = np.zeros((boot_n + 1, len(arches), len(budgets), len(scores), len(STRATEGIES)))
        nf = np.zeros_like(risks)
        legacy = pd.read_csv(ROOT / config["inputs"][cohort.lower() + "_locked_risk_curve"]["path"])
        for ai, arch in enumerate(arches):
            a = table.loc[table.architecture.eq(arch)].set_index("patient_id").loc[ids].reset_index()
            errors = a.abs_delta_rvef_points.to_numpy(float)
            flags, flag_order, individual, severity = flags_and_order(a.pred_edv_ml, a.pred_esv_ml, a.pred_rvef_points, ids)
            assert flags.sum() == config["expected_flags"][cohort][arch]
            if cohort == "MNMS":
                np.testing.assert_array_equal(a[FLAG_COLUMNS].to_numpy(bool), individual)
                np.testing.assert_array_equal(a.pathological_ratio_any.to_numpy(bool), flags)
                assert set(ids[flags]) == set(expected_flags.loc[expected_flags.architecture.eq(arch), "patient_id"])
            cases = a.loc[flags, ["cohort", "patient_id", "architecture", "pred_edv_ml", "pred_esv_ml", "pred_rvef_points", "abs_delta_rvef_points"]].copy()
            cases["severity"] = severity[flags]
            flagged_cases.extend(cases.to_dict("records"))
            for bi, budget in enumerate(budgets):
                k = int(np.floor(budget * n))
                sampled_flags = counts[:, flags].sum(axis=1)
                capacity.append(dict(cohort=cohort, architecture=arch, budget=budget, n=n, k=k,
                                     flags=int(flags.sum()), flags_reviewed=min(k, int(flags.sum())),
                                     flags_fraction_of_population=float(flags.mean()),
                                     review_capacity_consumed=min(k, int(flags.sum())) / k,
                                     secondary_places=max(0, k - int(flags.sum())),
                                     bootstrap_overflow_resamples=int((sampled_flags[1:] > k).sum()),
                                     bootstrap_no_secondary_resamples=int((sampled_flags[1:] >= k).sum()),
                                     bootstrap_max_flags=int(sampled_flags[1:].max())))
                for si, score in enumerate(scores):
                    values = a[score].to_numpy(float)
                    r, nonfinite, probabilities = evaluate(counts, errors, values, ids, flags, flag_order, k)
                    risks[:, ai, bi, si] = r
                    nf[:, ai, bi, si] = nonfinite
                    old = legacy.loc[legacy.row_type.eq("curve") & legacy.population.eq("all") &
                                     legacy.architecture.eq(arch) & legacy.qc_signal.eq(score) &
                                     legacy.error_target.eq("abs_delta_rvef_points") &
                                     np.isclose(legacy.coverage, (n-k)/n, rtol=0, atol=1e-12)]
                    assert len(old) == 1
                    lock_r = old.iloc[0]
                    for strategy, old_col in [("random", "random_risk"), ("qc", "detector_risk"), ("oracle", "oracle_risk")]:
                        diff = r[0, STRATEGIES.index(strategy)] - float(lock_r[old_col])
                        assert abs(diff) < 1e-10
                        checks.append(dict(cohort=cohort, architecture=arch, budget=budget, score=score,
                                           check="locked_curve_" + strategy, max_abs_difference=abs(diff), passed=True))
                    # Independent code-path check against the original risk implementation
                    # for every sample in zero-flag settings and selected samples otherwise.
                    positions = range(boot_n + 1) if not flags.any() else [0, 1, 2, 101, boot_n]
                    maxdiff = 0.
                    for ri in positions:
                        repeated = np.repeat(np.arange(n), counts[ri])
                        old_risk = library.retained_risk_curve(values[repeated], errors[repeated])[1][k]
                        maxdiff = max(maxdiff, abs(old_risk - r[ri, 1]))
                    assert maxdiff < 1e-10
                    checks.append(dict(cohort=cohort, architecture=arch, budget=budget, score=score,
                                       check="original_library_resamples", max_abs_difference=maxdiff, passed=True))
                    if not flags.any():
                        np.testing.assert_allclose(r[:, 0], r[:, 2], rtol=0, atol=1e-10)
                        np.testing.assert_array_equal(r[:, 1], r[:, 3])
                        checks.append(dict(cohort=cohort, architecture=arch, budget=budget, score=score,
                                           check="zero_flag_all_5000_resamples", max_abs_difference=float(np.max(abs(r[:, 0]-r[:, 2]))), passed=True))
                    for pi, patient_id in enumerate(ids):
                        for ti, strategy in enumerate(STRATEGIES):
                            selections.append(dict(cohort=cohort, architecture=arch, budget=budget, score=score,
                                                   patient_id=patient_id, strategy=strategy, k=k, flagged=bool(flags[pi]),
                                                   review_probability=float(probabilities[pi, ti])))
            print(f"{cohort} {arch}: policy counts, flags, and locked risk curves verified.", flush=True)
        assert np.isfinite(risks).all() and not nf.any()
        values = derived_metrics(risks)
        raw_arrays[cohort + "_metrics"] = values[1:]
        raw_arrays[cohort + "_point_metrics"] = values[0]
        raw_arrays[cohort + "_retained_nonfinite"] = nf[1:]
        for bi, budget in enumerate(budgets):
            for si, score in enumerate(scores):
                common = dict(cohort=cohort, budget=budget, score=score, n=n, k=int(np.floor(budget*n)))
                summary = dict(common)
                # Median of paired architecture-specific contrasts, never a
                # difference of separately summarized architecture medians.
                for mi, metric in enumerate(METRICS):
                    point, lo, hi = summarize(np.median(values[:, :, bi, si, mi], axis=1))
                    summary.update({metric + "_estimate": point, metric + "_ci_low": lo, metric + "_ci_high": hi})
                summary["retained_nonfinite_count_all_strategies"] = 0
                main_rows.append(summary)
                for ai, arch in enumerate(arches):
                    record = dict(common, architecture=arch)
                    for mi, metric in enumerate(METRICS):
                        point, lo, hi = summarize(values[:, ai, bi, si, mi])
                        record.update({metric + "_estimate": point, metric + "_ci_low": lo, metric + "_ci_high": hi})
                    record["retained_nonfinite_count_all_strategies"] = 0
                    architecture_rows.append(record)
    assert len(architecture_rows) == 64 and len(main_rows) == 16
    for name, records in [("main_results", main_rows), ("architecture_results", architecture_rows),
                          ("flag_capacity", capacity), ("patient_review_allocations", selections),
                          ("consistency_checks", checks), ("flagged_cases", flagged_cases)]:
        pd.DataFrame(records).to_csv(OUT / (name + ".csv"), index=False)
    np.savez_compressed(OUT / "bootstrap_patient_indices.npz", **sample_arrays)
    np.savez_compressed(OUT / "bootstrap_metrics.npz", **raw_arrays)
    write_json(OUT / "bootstrap_schema.json", dict(axes=["resample", "architecture", "budget", "score", "metric"],
               architectures=arches, budgets=budgets, scores=scores, metrics=METRICS,
               resample_count=boot_n, point_arrays="same axes without resample",
               patient_indices="zero-based columns index the stored cohort patient_ids; multiplicities retained",
               retained_nonfinite_axes=["resample", "architecture", "budget", "score", "strategy"], strategies=STRATEGIES))
    completed = datetime.now(timezone.utc)
    write_json(OUT / "verification.json", dict(status="PASS", exploratory=True, started_at_utc=started.isoformat(),
               completed_at_utc=completed.isoformat(), received_calendar_date=config["received_date"],
               due_calendar_date=config["due_date"], within_three_calendar_days=completed.date().isoformat() <= config["due_date"],
               architecture_comparisons=64, main_rows=16, bootstrap_replicates=boot_n,
               review_counts_checked_all_strategies_all_resamples=True, retained_nonfinite_count=0,
               nonfinite_audit="All required values finite; no exclusions, caps, penalties, or clipping applied",
               flag_cases_verified_against_locked_table=True, zero_flag_all_bootstraps_verified=True,
               python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__,
               platform=platform.platform(), seconds=(completed-started).total_seconds(),
               config_sha256=sha(HERE / "config.json"), analysis_script_sha256=sha(Path(__file__))))
    print("Completed: 64 paired architecture comparisons; 16 architecture-median summaries.", flush=True)


if __name__ == "__main__":
    run()
