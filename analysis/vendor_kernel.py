"""Preserved M&Ms subgroup statistic kernel."""
import numpy as np
from scipy import stats
from .protocol import SCORES
VARIABLES=SCORES+['dice_error','EDV','SV','EF']
SPECS=[('operating_point','Dice','none'),('operating_point','EDV','none'),('operating_point','EF','none')]
SPECS += [('rho_score_target',t,s) for s in SCORES for t in ['dice_error','EDV','SV','EF']]
SPECS += [('rho_dice_EF','EF','dice_oracle')]
SPECS += [('paired_gap','EF',s) for s in SCORES]
SPECS += [('EF_recovered_fraction','EF',s) for s in SCORES+['dice_oracle']]
def aurc(errors,order,idx,n):
    ranks=np.empty(len(errors),int);ranks[order]=np.arange(len(errors));valid=idx>=0;safe=np.maximum(idx,0)
    sampled_rank=np.where(valid,ranks[safe],len(errors)+1);positions=np.argsort(sampled_rank,axis=1,kind='stable')
    sampled_error=np.where(valid,errors[safe],0.);sorted_error=np.take_along_axis(sampled_error,positions,axis=1)
    tails=np.cumsum(sorted_error[:,::-1],axis=1)[:,::-1];den=n[:,None]-np.arange(idx.shape[1])[None,:]
    curve=np.divide(tails,den,out=np.zeros_like(tails),where=den>0)
    return curve.sum(axis=1)/n

def statistics(g,idx):
    safe=np.maximum(idx,0);valid=idx>=0;n=valid.sum(axis=1);assert (n>=3).all()
    x=g[VARIABLES].to_numpy(float).T;sampled=np.transpose(x[:,safe],(1,0,2));sampled=np.where(valid[:,None,:],sampled,np.nan)
    rr=stats.rankdata(sampled,axis=-1,method='average',nan_policy='omit');rr-=np.nanmean(rr,axis=-1,keepdims=True);norm=np.sqrt(np.nansum(rr*rr,axis=-1,keepdims=True))
    with np.errstate(invalid='ignore',divide='ignore'):rr=rr/norm
    constant=norm[:,:,0]==0;rr=np.nan_to_num(rr,nan=0.)
    r=np.einsum('bpn,bqn->bpq',rr,rr,optimize=False)
    for j in range(len(VARIABLES)):r[:,j,:][constant[:,j]]=np.nan;r[:,:,j][constant[:,j]]=np.nan
    values=[1-np.nanmean(sampled[:,4,:],axis=1),np.nanmedian(sampled[:,5,:],axis=1),np.nanmedian(sampled[:,7,:],axis=1)]
    for i in range(4):
        for j in [4,5,6,7]:values.append(r[:,i,j])
    values.append(r[:,4,7])
    for i in range(4):values.append(r[:,i,4]-r[:,i,7])
    errors=g.EF.to_numpy(float);ids=g.patient_id.to_numpy(str);oracle=aurc(errors,np.lexsort((ids,-errors)),idx,n)
    random=np.where(valid,errors[safe],0.).sum(axis=1)/n;den=random-oracle
    for policy in SCORES+['dice_oracle']:
        score=g.dice_error.to_numpy(float) if policy=='dice_oracle' else g[policy].to_numpy(float)
        detector=aurc(errors,np.lexsort((ids,-score)),idx,n)
        with np.errstate(invalid='ignore',divide='ignore'):rf=(random-detector)/den
        rf[np.abs(den)<=np.finfo(float).eps]=np.nan;values.append(rf)
    return np.column_stack(values)
