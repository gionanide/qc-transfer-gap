"""Preserved tolerance kernel; budget-review primitives come from the existing repo."""
import numpy as np
from scipy import stats
from cardiac_functional_qc import _locked_review
from .protocol import SCORES,BUDGETS,ENDPOINTS,ROBUST,GOALS
PAPER={'flags_and_order':_locked_review.flags_and_order,'take_ranked':_locked_review.take_ranked}
POLICIES=['random']+SCORES+['dice_oracle','plausibility_random']+['plausibility_'+s for s in SCORES]+['true_error_oracle']
WEIGHT_CACHE={}
def make_context(g):
    ids=g.patient_id.to_numpy(str);assert list(ids)==sorted(ids)
    errors=g.EF.to_numpy(float);relative=g.relative_EDV_error.to_numpy(float)
    flags,flag_order,_,_=PAPER['flags_and_order'](g.pred_edv_ml,g.pred_esv_ml,g.pred_ef,ids)
    orders={s:np.lexsort((ids,-g[s].to_numpy(float))) for s in SCORES}
    orders['dice_oracle']=np.lexsort((ids,-g.dice_error.to_numpy(float)))
    for score in SCORES:orders['plausibility_'+score]=np.r_[flag_order,orders[score][~flags[orders[score]]]]
    orders['true_error_oracle']=np.lexsort((ids,-errors))
    return dict(ids=ids,errors=errors,relative=relative,flags=flags,flag_order=flag_order,orders=orders,edv_oracle=np.lexsort((ids,-relative)),error_order=np.lexsort((ids,errors)),bad=np.column_stack([errors>5,errors>10,relative>.1]).astype(np.int16))

def retained_counts(ctx,counts,k):
    n=counts.sum(axis=1);take=PAPER['take_ranked'];flags=ctx['flags'];flag_reviewed=take(counts,ctx['flag_order'],k);secondary=k-flag_reviewed.sum(axis=1)
    result={'random':counts*((n-k)/n)[:,None]}
    for policy,order in ctx['orders'].items():result[policy]=counts-take(counts,order,k)
    unflag_n=counts[:,~flags].sum(axis=1);fraction=np.divide(secondary,unflag_n,out=np.zeros(len(n),float),where=unflag_n>0)
    pr=(counts-flag_reviewed).astype(float);pr[:,~flags]*=(1-fraction[:,None]);result['plausibility_random']=pr
    for a in result.values():np.testing.assert_allclose(a.sum(axis=1),n-k,rtol=0,atol=1e-10)
    edv_oracle=counts-take(counts,ctx['edv_oracle'],k)
    return result,edv_oracle

def deterministic_robust(ctx,retained):
    counts=retained[:,ctx['error_order']];errors=ctx['errors'][ctx['error_order']]
    cumulative=np.cumsum(counts,axis=1);n=cumulative[:,-1];before=cumulative-counts
    assert (n>0).all() and np.all(counts==np.floor(counts))
    lo=(n-1)//2+1;hi=n//2+1
    jlo=np.argmax(cumulative>=lo[:,None],axis=1);jhi=np.argmax(cumulative>=hi[:,None],axis=1)
    median=(errors[jlo]+errors[jhi])/2
    trim=np.floor(.1*n).astype(int)
    kept=np.maximum(0,np.minimum(cumulative,(n-trim)[:,None])-np.maximum(before,trim[:,None]))
    trimmed=(kept@errors)/(n-2*trim)
    return np.column_stack([(counts@errors)/n,median,trimmed])

def random_order_stat_prefix(N,R):
    """Expectation over every equally likely exact-R subset of N occurrences."""
    key=(int(N),int(R))
    if key in WEIGHT_CACHE:return WEIGHT_CACHE[key]
    N,R=key;assert 1<=R<=N
    if R==N:
        median=np.zeros(N);median[(R-1)//2]+=.5;median[R//2]+=.5
        t=int(np.floor(.1*R));trimmed=np.zeros(N);trimmed[t:R-t]=1/(R-2*t)
    else:
        successes=np.arange(N);jlo=(R-1)//2+1;jhi=R//2+1
        median=R/N*.5*(stats.hypergeom.pmf(jlo-1,N-1,successes,R-1)+stats.hypergeom.pmf(jhi-1,N-1,successes,R-1))
        t=int(np.floor(.1*R))
        probability=stats.hypergeom.cdf(R-t-1,N-1,successes,R-1)-stats.hypergeom.cdf(t-1,N-1,successes,R-1)
        trimmed=R/N*probability/(R-2*t)
    np.testing.assert_allclose([median.sum(),trimmed.sum()],[1,1],rtol=0,atol=2e-12)
    prefixes=np.column_stack([np.r_[0,np.cumsum(median)],np.r_[0,np.cumsum(trimmed)]])
    WEIGHT_CACHE[key]=prefixes
    return prefixes

def expected_random_robust(ctx,population_counts,R):
    counts=population_counts[:,ctx['error_order']];errors=ctx['errors'][ctx['error_order']];cumulative=np.cumsum(counts,axis=1).astype(int);before=cumulative-counts;N=cumulative[:,-1]
    out=np.empty((len(counts),3));out[:,0]=(counts@errors)/N
    for Nvalue,Rvalue in set(zip(N.tolist(),np.asarray(R).tolist())):
        ix=np.flatnonzero((N==Nvalue)&(R==Rvalue));prefix=random_order_stat_prefix(Nvalue,Rvalue)
        weights=prefix[cumulative[ix]]-prefix[before[ix]]
        out[ix,1:]=np.einsum('bnj,n->bj',weights,errors,optimize=False)
    return out

def minimum_review(counts,order,bad,denominator):
    """First k satisfying bad_remaining*denominator <= n-k, while retaining >=1."""
    n=counts.sum(axis=1);bad_total=counts@bad
    result=np.where(bad_total*denominator<=n,0.,np.inf)
    if not len(order):return result
    available=counts[:,order];before=np.cumsum(available,axis=1)-available;bad_values=bad[order]
    removed_before=np.cumsum(available*bad_values,axis=1)-available*bad_values
    bad_before=bad_total[:,None]-removed_before;remaining_before=n[:,None]-before
    numerator=denominator*bad_before-remaining_before
    u=np.maximum(0,(numerator+(denominator-2))//(denominator-1))
    possible=((bad_values[None,:]==1)|(numerator<=0))&(u<=available)
    candidate=before+u;possible&=candidate<n[:,None]
    smallest=np.min(np.where(possible,candidate,np.inf),axis=1)
    return np.minimum(result,smallest)/n

def specifications(boot=False):
    budgets=[.1,.2] if boot else BUDGETS
    result=[('exceedance',p,b,e) for p in POLICIES for b in budgets for e in ENDPOINTS]
    result += [('robust',p,b,metric) for p in POLICIES for b in [.1,.2] for metric in ROBUST]
    result += [('required_budget',p,e,den) for p in POLICIES for e,den in GOALS]
    return result

def evaluate(ctx,counts,boot=False):
    assert (counts>=0).all();n=counts.sum(axis=1);assert (n>0).all()
    output={};budgets=[.1,.2] if boot else BUDGETS
    for budget in budgets:
        k=np.floor(budget*n).astype(int);R=n-k
        retained,edv_oracle=retained_counts(ctx,counts,k)
        for policy in POLICIES:
            weights=retained[policy]
            proportions=(weights@ctx['bad'])/R[:,None]
            if policy=='true_error_oracle':proportions[:,2]=(edv_oracle@ctx['bad'][:,2])/R
            for j,endpoint in enumerate(ENDPOINTS):output[('exceedance',policy,budget,endpoint)]=proportions[:,j]
            if budget in [.1,.2]:
                if policy=='random':robust=expected_random_robust(ctx,counts,R)
                elif policy=='plausibility_random':
                    flags_n=counts[:,ctx['flags']].sum(axis=1);random_branch=k>=flags_n;robust=np.empty((len(counts),3))
                    if (~random_branch).any():robust[~random_branch]=deterministic_robust(ctx,weights[~random_branch])
                    if random_branch.any():
                        population=counts[random_branch].copy();population[:,ctx['flags']]=0
                        robust[random_branch]=expected_random_robust(ctx,population,R[random_branch])
                else:robust=deterministic_robust(ctx,weights)
                for j,metric in enumerate(ROBUST):output[('robust',policy,budget,metric)]=robust[:,j]
    for policy in POLICIES:
        for endpoint,den in GOALS:
            bad=ctx['bad'][:,ENDPOINTS.index(endpoint)]
            if policy=='random':required=np.where((counts@bad)*den<=n,0.,np.inf)
            else:
                order=ctx['flag_order'] if policy=='plausibility_random' else ctx['orders'][policy]
                required=minimum_review(counts,order,bad,den)
            output[('required_budget',policy,endpoint,den)]=required
    return output
