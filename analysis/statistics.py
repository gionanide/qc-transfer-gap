"""Original rank, partial-correlation and RF kernels; no patient data."""
import numpy as np
from scipy import stats
from . import risk_coverage as RISK
def rank_correlations(x):
    # x shape (batch, variables, patients); ranks are recomputed for each resample.
    rr=stats.rankdata(x,axis=-1,method='average');rr-=rr.mean(axis=-1,keepdims=True)
    norm=np.sqrt(np.sum(rr*rr,axis=-1,keepdims=True))
    with np.errstate(invalid='ignore',divide='ignore'):rr=rr/norm
    return np.einsum('bpn,bqn->bpq',rr,rr,optimize=False)

def partial(r,x,y,z):
    den=(1-r[:,x,z]**2)*(1-r[:,y,z]**2)
    with np.errstate(invalid='ignore',divide='ignore'):
        result=(r[:,x,y]-r[:,x,z]*r[:,y,z])/np.sqrt(den)
    result[den<=1e-14]=np.nan
    return np.clip(result,-1,1)
def rf_occurrences(scores,error):
    """Rows must already be ordered by patient ID, then original draw index."""
    n=len(error);p=scores.shape[1]
    if n==0:return np.full(p,np.nan),np.nan,np.nan,np.nan,np.full(p,np.nan)
    _,rc=RISK.random_risk_curve(error)
    _,oc=RISK.oracle_risk_curve(error)
    random_auc=RISK.discrete_aurc(rc);oracle_auc=RISK.discrete_aurc(oc);den=random_auc-oracle_auc
    auc=np.empty(p);rf=np.empty(p)
    for j in range(p):
        _,curve=RISK.retained_risk_curve(scores[:,j],error)
        auc[j]=RISK.discrete_aurc(curve)
        value,status=RISK.normalized_deferral_gain(auc[j],random_auc,oracle_auc)
        # The unchanged paper guard is abs(den)<=machine epsilon. Also reject
        # a negative denominator, as requested by the below-guard rule.
        rf[j]=value if status=='defined' and den>np.finfo(np.float64).eps else np.nan
    return rf,random_auc,oracle_auc,den,auc

def median_estimands(rf):
    """Ordinary medians propagate undefined architecture values; never nanmedian."""
    median_rf=np.median(rf,axis=1)
    difference=rf[:,:,4,None]-rf[:,:,:4]
    median_difference=np.median(difference,axis=1)
    difference_of_medians=median_rf[:,4,None]-median_rf[:,:4]
    return median_rf,difference,median_difference,difference_of_medians
def finite_interval(values):
    a=np.asarray(values);v=a[np.isfinite(a)];lo,hi=np.quantile(v,[.025,.975],method='linear') if len(v) else (np.nan,np.nan)
    return dict(ci_low=float(lo),ci_high=float(hi),bootstrap_replicates=len(a),bootstrap_valid=len(v),bootstrap_undefined=len(a)-len(v))

def extended_interval(values):
    a=np.asarray(values);assert not np.isnan(a).any() and not np.isneginf(a).any();a=np.sort(a)
    def quantile(p):
        h=(len(a)-1)*p;i=int(np.floor(h));j=int(np.ceil(h));fraction=h-i
        if fraction==0:return float(a[i])
        if np.isposinf(a[j]):return float('inf')
        return float(a[i]+fraction*(a[j]-a[i]))
    return dict(ci_low=quantile(.025),ci_high=quantile(.975),bootstrap_replicates=len(a),bootstrap_valid=len(a),bootstrap_undefined=0,bootstrap_attained=int(np.isfinite(a).sum()),bootstrap_unattained=int(np.isposinf(a).sum()))
def describe(spec):
    kind,policy,a,b=spec
    if kind=='exceedance':return dict(analysis=kind,policy=policy,budget=float(a),endpoint=b,metric='retained_exceedance_proportion')
    if kind=='robust':return dict(analysis=kind,policy=policy,budget=float(a),endpoint='EF',metric=b)
    return dict(analysis=kind,policy=policy,endpoint=a,metric='minimum_review_fraction',maximum_exceedance_fraction=1/b)

def counts_from_master(sampled,mapping,n):
    local=mapping[sampled];valid=local>=0;counts=np.zeros((len(local),n),np.int16)
    row=np.broadcast_to(np.arange(len(local))[:,None],local.shape)
    np.add.at(counts,(row[valid],local[valid]),1)
    return counts
