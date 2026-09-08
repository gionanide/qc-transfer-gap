"""Small interface and policy-invariant tests; no additional study experiments."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardiac_functional_qc import _locked_review as locked
from cardiac_functional_qc.review_adapter import InputError, load_config, run, validate_input

ROOT = Path(__file__).resolve().parents[1]


def example():
    return pd.read_csv(ROOT / "examples/example_review_input.csv", float_precision="round_trip")


def validate(frame, tmp_path):
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False, float_format="%.17g")
    return validate_input(path, load_config(ROOT / "configs/example_review.yaml"), tmp_path)


def test_finite_implausibility_is_retained_and_undefined_is_rejected(tmp_path):
    d = validate(example(), tmp_path)
    assert len(d) == 10 and d.pred_rvef_points.min() < 0
    assert d.abs_delta_rvef_points.max() > 50
    d.loc[0, "pred_rvef_points"] = np.nan
    with pytest.raises(InputError, match="Nonfinite"):
        validate(d, tmp_path)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "units", "direction", "flag", "membership", "reference", "signed_error"])
def test_invalid_inputs_fail_explicitly(kind, tmp_path):
    d = example()
    if kind in ["units", "direction"]:
        import yaml
        c = load_config(ROOT / "configs/example_review.yaml")
        if kind == "units":
            c["units"]["ef"] = "fraction"
        else:
            c["scores"][0]["direction"] = "auto"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(c))
        with pytest.raises(InputError):
            load_config(p)
        return
    if kind == "duplicate":
        d = pd.concat([d, d.iloc[:1]], ignore_index=True)
    if kind == "missing":
        d = d.drop(columns="example_qc")
    if kind == "flag":
        d["pathological_ratio_any"] = False
    if kind == "membership":
        d = pd.concat([d, d.iloc[:-1].assign(architecture="other_model")], ignore_index=True)
    if kind == "reference":
        d = pd.concat([d, d.assign(architecture="other_model", ref_rvef_points=d.ref_rvef_points+1)], ignore_index=True)
    if kind == "signed_error":
        d["delta_rvef_points"] = 999.
    with pytest.raises(InputError):
        validate(d, tmp_path)


def test_deployable_rankings_do_not_consult_reference_errors():
    d = example()
    ids = d.patient_id.to_numpy(str)
    f, order, _, _ = locked.flags_and_order(d.pred_edv_ml,d.pred_esv_ml,d.pred_rvef_points,ids)
    counts = np.ones((1,len(d)),int)
    score = d.example_qc.to_numpy()
    a = locked.evaluate(counts,np.arange(10,dtype=float),score,ids,f,order,2)[2]
    b = locked.evaluate(counts,np.arange(10,dtype=float)[::-1]*1000,score,ids,f,order,2)[2]
    np.testing.assert_array_equal(a[:,:4],b[:,:4])
    assert a[1,1] == 1  # highest score
    assert a[:,3].sum()==2 and a[1,3]==1  # flag consumes one of two slots


def test_arithmetic_and_median_of_contrasts():
    risks = np.array([[0.,0.,1.,0.,0.],[0.,0.,3.,2.,0.],[0.,0.,5.,100.,0.],[0.,0.,100.,101.,0.]])
    m=locked.derived_metrics(risks)
    assert np.median(m[:,5]) != np.median(risks[:,2])-np.median(risks[:,3])
    np.testing.assert_array_equal(m[:,6],risks[:,1]-risks[:,2])


def test_one_model_is_one_model_and_reference_volume_adapter(tmp_path):
    d=example()
    d["ref_edv_ml"]=100.
    d["ref_esv_ml"]=100-d.ref_rvef_points
    d=d.drop(columns="ref_rvef_points")
    p=tmp_path/"volume_reference.csv"
    d.to_csv(p,index=False,float_format="%.17g")
    output=tmp_path/"result"
    run(p,ROOT/"configs/example_review.yaml",output)
    results=pd.read_csv(output/"main_results.csv")
    arch=pd.read_csv(output/"architecture_results.csv")
    assert len(results)==2 and len(arch)==2
    assert results.n_architectures.eq(1).all()
    assert results.n.eq(10).all()
    assert results.k.tolist()==[1,2]
    np.testing.assert_allclose(results.delta1_estimate,arch.delta1_estimate)
    with np.load(output/"bootstrap_metrics_and_indices.npz") as z:
        draws=z["SYNTHETIC_indices"]
        assert draws.shape==(5000,10)
        assert len(set(draws[0]))<10
        risk=z["SYNTHETIC_metrics"][...,:5]
        assert np.all(risk[...,4,None] <= risk+1e-10)
