"""Input validation and thin orchestration around the byte-identical review kernel."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import _locked_review as locked
from .propagation import ef_points

KEYS = ["cohort", "patient_id", "architecture", "ventricle"]
PRED = ["pred_edv_ml", "pred_esv_ml", "pred_rvef_points"]
FLAG_COLUMNS = locked.FLAG_COLUMNS + ["pathological_ratio_any"]


class InputError(ValueError):
    """An unsupported input is rejected before any policy result is emitted."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(c, dict) or c.get("schema_version") != 1:
        raise InputError("Expected schema_version: 1")
    if c.get("units") != {"volume": "mL", "ef": "percentage_points"}:
        raise InputError("Declare volume: mL and ef: percentage_points; no automatic unit conversion")
    if c.get("ventricle") != "RV":
        raise InputError("This v1 review interface is RV-only; legacy LV results are documented separately")
    scores = c.get("scores", [])
    if not scores or any(not isinstance(s, dict) or s.get("direction") != "higher" or not isinstance(s.get("column"), str) for s in scores):
        raise InputError("Declare score columns with direction: higher; direction is never learned or flipped")
    names = [s["column"] for s in scores]
    if len(set(names)) != len(names) or set(names) & set(KEYS + PRED + ["ref_rvef_points", "abs_delta_rvef_points"]):
        raise InputError("Score identifiers must be unique and distinct from measurements and keys")
    if c.get("budgets") != [.1, .2] or c.get("bootstrap_replicates") != 5000:
        raise InputError("The paper protocol uses budgets [0.1, 0.2] and exactly 5000 patient resamples")
    if not isinstance(c.get("cohort_seeds"), dict):
        raise InputError("Declare a fixed integer seed for each cohort")
    return c


def validate_input(path, config, output):
    scores = [s["column"] for s in config["scores"]]
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    if len(header) != len(set(header)):
        raise InputError("Duplicate CSV column names")
    d = pd.read_csv(path, dtype={k: str for k in KEYS}, keep_default_na=False, float_precision="round_trip")
    if "ventricle" not in d:
        d["ventricle"] = config["ventricle"]
    required = KEYS + PRED + scores
    missing = sorted(set(required)-set(d.columns))
    if missing:
        raise InputError("Missing required columns: " + ", ".join(missing))
    if d.empty or d[KEYS].isna().any().any() or d[KEYS].apply(lambda s: s.str.strip().eq("")).any().any():
        raise InputError("Empty dataset or missing key values")
    if d.duplicated(KEYS).any():
        raise InputError("Duplicate (cohort, patient_id, architecture, ventricle) records")
    if not d.ventricle.eq("RV").all():
        raise InputError("Mixed/non-RV records are unsupported by this RV-only entry point")
    reference = [c for c in ["ref_rvef_points", "ref_edv_ml", "ref_esv_ml"] if c in d]
    if "ref_rvef_points" not in reference and not {"ref_edv_ml", "ref_esv_ml"}.issubset(reference):
        raise InputError("Supply reference EF or both reference EDV and ESV")
    optional = [c for c in ["pred_sv_ml", "ref_sv_ml", "delta_rvef_points", "abs_delta_rvef_points", "rv_dice_ed", "rv_dice_es", "rv_dice_mean"] if c in d]
    numeric = PRED + scores + reference + optional
    audit = []
    for field in numeric:
        original = d[field]
        d[field] = pd.to_numeric(original, errors="coerce")
        x = d[field].to_numpy(float)
        audit.append(dict(field=field, records=len(d), missing_or_nan=int(np.isnan(x).sum()),
                          positive_infinite=int(np.isposinf(x).sum()), negative_infinite=int(np.isneginf(x).sum())))
    dump(output / "nonfinite_audit.json", audit)
    # Preserve and audit the historical flag rules before rejecting unsupported undefined EF.
    flags, _, individual, severity = locked.flags_and_order(d.pred_edv_ml, d.pred_esv_ml, d.pred_rvef_points, d.patient_id)
    derived = pd.DataFrame(individual, columns=locked.FLAG_COLUMNS)
    derived["pathological_ratio_any"] = flags
    audit_records = d[KEYS].copy()
    for col in FLAG_COLUMNS:
        audit_records[col] = derived[col].to_numpy()
        if col in d:
            text = d[col].astype(str).str.lower()
            if not text.isin(["true", "false", "1", "0"]).all():
                raise InputError(f"Stored flag {col} must be explicit true/false or 1/0")
            if not np.array_equal(text.isin(["true", "1"]).to_numpy(), derived[col].to_numpy()):
                raise InputError(f"Stored flag disagrees with prediction-only rule: {col}")
    audit_records["severity"] = severity
    audit_records["flag_reasons"] = [";".join(col for col in locked.FLAG_COLUMNS if derived.loc[i, col]) for i in range(len(d))]
    audit_records.to_csv(output / "plausibility_audit.csv", index=False)
    if any(x["missing_or_nan"]+x["positive_infinite"]+x["negative_infinite"] for x in audit):
        raise InputError("Nonfinite/malformed required numeric input. See audit; v1 does not exclude, penalize, or assign finite MAE to undefined observations")
    if d.pred_edv_ml.le(0).any():
        raise InputError("Nonpositive predicted EDV implies undefined volume-derived EF; audited but unsupported in v1")
    if not np.allclose(d.pred_rvef_points, ef_points(d.pred_edv_ml, d.pred_esv_ml), rtol=0, atol=1e-10):
        raise InputError("Predicted EF is inconsistent with volume-derived EF in percentage points")
    if {"ref_edv_ml", "ref_esv_ml"}.issubset(d.columns):
        computed = ef_points(d.ref_edv_ml, d.ref_esv_ml)
        if not np.isfinite(computed).all():
            raise InputError("Undefined reference EF from supplied reference volumes")
        if "ref_rvef_points" in d and not np.allclose(d.ref_rvef_points, computed, rtol=0, atol=1e-10):
            raise InputError("Reference EF disagrees with reference phase volumes")
        if "ref_rvef_points" not in d:
            d["ref_rvef_points"] = computed
    error = abs(d.pred_rvef_points-d.ref_rvef_points)
    if "delta_rvef_points" in d and not np.allclose(d.delta_rvef_points, d.pred_rvef_points-d.ref_rvef_points, rtol=0, atol=1e-10):
        raise InputError("Stored signed EF error disagrees with prediction/reference EF")
    if "abs_delta_rvef_points" in d:
        if not np.allclose(d.abs_delta_rvef_points, error, rtol=0, atol=1e-10):
            raise InputError("Stored absolute EF error disagrees with prediction/reference EF")
    else:
        d["abs_delta_rvef_points"] = error
    if {"rv_dice_ed", "rv_dice_es", "rv_dice_mean"}.issubset(d.columns):
        if not np.allclose(d.rv_dice_mean, (d.rv_dice_ed+d.rv_dice_es)/2, rtol=0, atol=1e-10):
            raise InputError("Stored Dice mean differs from arithmetic ED/ES mean")
    alignment = []
    for cohort, group in d.groupby("cohort", sort=True):
        seed = config["cohort_seeds"].get(cohort)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise InputError(f"Missing valid fixed integer seed for cohort {cohort}")
        members = [set(g.patient_id) for _, g in group.groupby("architecture")]
        if any(m != members[0] for m in members[1:]):
            raise InputError(f"Patient membership differs across models in {cohort}; no silent intersection")
        if "architectures" in config and set(group.architecture) != set(config["architectures"]):
            raise InputError(f"Model identifiers differ from configured paper set in {cohort}")
        if "expected_cohort_sizes" in config and len(members[0]) != config["expected_cohort_sizes"].get(cohort):
            raise InputError("Patient count differs from the locked paper population")
        for col in ["ref_rvef_points", "ref_edv_ml", "ref_esv_ml"]:
            if col in group and group.groupby("patient_id")[col].agg(lambda x: np.ptp(x.to_numpy())).max() > 1e-10:
                raise InputError(f"Reference measurements disagree across models: {cohort}, {col}")
        alignment.append(dict(cohort=cohort, unique_patients=len(members[0]), models=len(members), records=len(group), score_values=len(group)*len(scores), excluded=0))
    dump(output / "input_alignment.json", alignment)
    return d


def compare_references(output, config, config_dir):
    report = []
    for name, keys in [("main_results.csv", ["cohort","budget","score"]),
                       ("architecture_results.csv", ["cohort","budget","score","architecture"])]:
        expected = pd.read_csv(config_dir / config["expected"] / name)
        actual = pd.read_csv(output / name)
        expected = expected.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)
        if len(expected) != len(actual) or not expected[keys].equals(actual[keys]):
            raise InputError("Locked result keys do not match: " + name)
        columns = [c for c in expected if c.endswith(("_estimate","_ci_low","_ci_high"))]
        difference = np.max(np.abs(actual[columns].to_numpy()-expected[columns].to_numpy()))
        passed = bool(np.allclose(actual[columns], expected[columns], rtol=0, atol=config["absolute_tolerance"]))
        report.append(dict(file=name, rows=len(actual), numerical_columns=len(columns), maximum_absolute_difference=float(difference), tolerance=config["absolute_tolerance"], passed=passed))
    printed = pd.read_csv(config_dir / config["printed_reference"])
    actual = pd.read_csv(output / "main_results.csv").query("cohort == 'MNMS'").reset_index(drop=True)
    columns = [c for c in printed if c not in ["cohort", "budget", "score"]]
    printed_pass = all(f"{a:.2f}" == f"{b:.2f}" for a,b in zip(actual[columns].to_numpy().ravel(),printed[columns].to_numpy().ravel()))
    report.append(dict(file="Table 4 printed values", rows=8, passed=printed_pass))
    dump(output / "reference_comparison.json", report)
    if not all(x["passed"] for x in report):
        raise InputError("Replay differs from locked or printed reference; see comparison report")


def run(input_path, config_path, output_path, paper=False):
    start = time.perf_counter()
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise InputError("Output directory must be new or empty, so stale success results cannot survive a failed run")
    output.mkdir(parents=True, exist_ok=True)
    try:
        c = load_config(config_path)
        input_path = Path(input_path) if input_path else config_path.parent / c["input"]
        if paper:
            if sha(input_path) != c["input_sha256"]:
                raise InputError("Paper input SHA-256 mismatch")
            manifest = json.loads((config_path.parent / c["source_manifest"]).read_text())
            if sha(Path(locked.__file__)) != manifest["locked_review_kernel_sha256"]:
                raise InputError("Locked review implementation SHA-256 mismatch")
        d = validate_input(input_path, c, output)
        saved = None
        if paper:
            path = config_path.parent / c["saved_bootstrap_indices"]
            if sha(path) != c["indices_sha256"]:
                raise InputError("Saved bootstrap-index SHA-256 mismatch")
            saved = np.load(path, allow_pickle=False)
        scores = [s["column"] for s in c["scores"]]
        arch_rows, main_rows, capacities, allocations, bootstrap = [], [], [], [], {}
        for cohort, group in d.groupby("cohort", sort=True):
            ids = np.array(sorted(group.patient_id.unique()), dtype=str)
            n = len(ids)
            arches = c.get("architectures", sorted(group.architecture.unique()))
            indices = np.random.default_rng(c["cohort_seeds"][cohort]).integers(0,n,size=(5000,n),dtype=np.int32)
            if saved is not None:
                np.testing.assert_array_equal(ids, saved[cohort+"_patient_ids"])
                np.testing.assert_array_equal(indices, saved[cohort+"_indices"])
                indices = saved[cohort+"_indices"]
            counts = np.zeros((5001,n),dtype=np.int32)
            counts[0] = 1
            np.add.at(counts[1:], (np.arange(5000)[:,None], indices), 1)
            assert np.all(counts.sum(axis=1)==n)
            risks = np.zeros((5001,len(arches),2,len(scores),5))
            for ai, arch in enumerate(arches):
                a = group.loc[group.architecture.eq(arch)].set_index("patient_id").loc[ids]
                flags, order, _, _ = locked.flags_and_order(a.pred_edv_ml,a.pred_esv_ml,a.pred_rvef_points,ids)
                errors = a.abs_delta_rvef_points.to_numpy(float)
                for bi,budget in enumerate(c["budgets"]):
                    k = int(np.floor(budget*n))
                    flag_n = counts[:,flags].sum(axis=1)
                    capacities.append(dict(cohort=cohort,architecture=arch,budget=budget,n=n,k=k,retained=n-k,flags=int(flags.sum()),
                        flags_reviewed=min(k,int(flags.sum())),review_capacity_consumed=min(k,int(flags.sum()))/k if k else 0.,secondary_places=max(0,k-int(flags.sum())),
                        bootstrap_overflow_resamples=int((flag_n[1:]>k).sum()),bootstrap_no_secondary_resamples=int((flag_n[1:]>=k).sum()),bootstrap_max_flags=int(flag_n[1:].max())))
                    for si,score in enumerate(scores):
                        r,nf,probabilities=locked.evaluate(counts,errors,a[score].to_numpy(float),ids,flags,order,k)
                        assert np.isfinite(r).all() and not nf.any()
                        if not np.all(r[:,4,None] <= r+1e-10):
                            raise AssertionError("Unconstrained oracle exceeds another policy's retained error")
                        if not flags.any():
                            np.testing.assert_allclose(r[:,0],r[:,2],rtol=0,atol=1e-10)
                            np.testing.assert_array_equal(r[:,1],r[:,3])
                        risks[:,ai,bi,si]=r
                        for pi,patient in enumerate(ids):
                            for ti,strategy in enumerate(locked.STRATEGIES):
                                allocations.append(dict(cohort=cohort,architecture=arch,budget=budget,score=score,patient_id=patient,
                                    policy=strategy,review_probability=float(probabilities[pi,ti]),non_deployable=strategy=="oracle"))
            metrics=locked.derived_metrics(risks)
            bootstrap[cohort+"_metrics"]=metrics[1:]
            bootstrap[cohort+"_indices"]=indices
            bootstrap[cohort+"_patient_ids"]=ids
            for bi,budget in enumerate(c["budgets"]):
                for si,score in enumerate(scores):
                    common=dict(cohort=cohort,budget=budget,score=score,n=n,k=int(np.floor(budget*n)))
                    row=dict(common)
                    for mi,metric in enumerate(locked.METRICS):
                        for suffix,value in zip(["_estimate","_ci_low","_ci_high"],locked.summarize(np.median(metrics[:,:,bi,si,mi],axis=1))):
                            row[metric+suffix]=value
                    row["retained_nonfinite_count_all_strategies"]=0
                    row["n_architectures"]=len(arches)
                    main_rows.append(row)
                    for ai,arch in enumerate(arches):
                        row=dict(common,architecture=arch)
                        for mi,metric in enumerate(locked.METRICS):
                            for suffix,value in zip(["_estimate","_ci_low","_ci_high"],locked.summarize(metrics[:,ai,bi,si,mi])):
                                row[metric+suffix]=value
                        row["retained_nonfinite_count_all_strategies"]=0
                        arch_rows.append(row)
            dump(output/(cohort+"_bootstrap_schema.json"),dict(axes=["resample","architecture","budget","score","metric"],
                 architectures=arches,budgets=c["budgets"],scores=scores,metrics=locked.METRICS,duplicate_patient_occurrences_preserved=True))
        for filename,rows in [("main_results.csv",main_rows),("architecture_results.csv",arch_rows),("flag_capacity.csv",capacities),("patient_review_allocations.csv",allocations)]:
            pd.DataFrame(rows).to_csv(output/filename,index=False,float_format="%.17g",lineterminator="\n")
        np.savez_compressed(output/"bootstrap_metrics_and_indices.npz",**bootstrap)
        # Export all five policies and uncertainty explicitly, including the oracle label.
        long=[]
        for row in arch_rows:
            for policy in locked.STRATEGIES:
                long.append({**{k:row[k] for k in ["cohort","architecture","budget","score","n","k"]},"policy":policy,
                    "retained_n":row["n"]-row["k"],"non_deployable":policy=="oracle","expected_random_policy":policy in ["random","plausibility_random"],
                    "retained_nonfinite_count":0,**{s:row[policy+"_"+s] for s in ["estimate","ci_low","ci_high"]}})
        pd.DataFrame(long).to_csv(output/"policy_risks.csv",index=False,float_format="%.17g")
        if paper:
            compare_references(output,c,config_path.parent)
        display=pd.DataFrame(main_rows)
        if paper:
            display=display.loc[display.cohort.eq("MNMS")]
        lines=["**Exploratory budget-matched RV EF review; unadjusted 95% paired bootstrap intervals**", "",
            "| Cohort | Budget | Score | Random | P + random | P + QC | Oracle (non-deployable) | Delta1 [95% CI] |",
            "|---|---:|---|---:|---:|---:|---:|---:|"]
        for _,r in display.iterrows():
            lines.append(f"| {r.cohort} | {r.budget:.0%} | {r.score} | {r.random_estimate:.2f} | {r.plausibility_random_estimate:.2f} | {r.plausibility_qc_estimate:.2f} | {r.oracle_estimate:.2f} | {r.delta1_estimate:.2f} [{r.delta1_ci_low:.2f}, {r.delta1_ci_high:.2f}] |")
        lines += ["", "Risks and contrasts are in EF percentage points. Model medians summarize the actual supplied models; one model remains one model. Contrasts precede their median. QC-alone risks and Delta2 are in the CSVs. Positive Delta1 favours adding QC; negative Delta2 favours QC alone over P + random. No inference of equivalence or successful correction is made."]
        (output/"Table4.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        metadata=dict(status="PASS",exploratory=True,source_input_sha256=sha(input_path),config_sha256=sha(config_path),
            locked_kernel_sha256=sha(Path(locked.__file__)),numpy=np.__version__,pandas=pd.__version__,pyyaml=yaml.__version__,
            runtime_seconds=time.perf_counter()-start,bootstrap_replicates=5000,cohort_seeds=c["cohort_seeds"],
            saved_indices_verified=paper,raw_images_or_checkpoints_required=False,excluded_patients=0)
        dump(output/"run_manifest.json",metadata)
        print(f"PASS: {len(arch_rows)} architecture comparisons; {metadata['runtime_seconds']:.3f} seconds")
    except (InputError, ValueError, AssertionError, KeyError) as error:
        dump(output/"validation_error.json",dict(status="ERROR",reason=str(error)))
        raise
