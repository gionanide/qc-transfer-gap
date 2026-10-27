"""Exact Step 5 score-evaluation statistics with configurable spread-score names."""
import numpy as np
from scipy import stats
from .statistics import partial
from .vendor_kernel import aurc
NEW_SCORES=[]
STAT_SPECS=[]
def dice_dependent(spec):return spec[0] in ['paired_gap','partial_rho_score_EF_given_dice','rho_dice_EF'] or spec[1]=='dice_error' or spec[2]=='dice_oracle'

def stat_values(g,idx,has_dice):
    valid=idx>=0;safe=np.maximum(idx,0);n=valid.sum(axis=1);assert (n>=3).all()
    columns=NEW_SCORES+(['dice_error'] if has_dice else [])+['EDV','SV','EF'];lookup={c:i for i,c in enumerate(columns)}
    x=g[columns].to_numpy(float).T;sample=np.transpose(x[:,safe],(1,0,2));sample=np.where(valid[:,None,:],sample,np.nan)
    ranked=stats.rankdata(sample,axis=-1,method='average',nan_policy='omit');ranked-=np.nanmean(ranked,axis=-1,keepdims=True);norm=np.sqrt(np.nansum(ranked*ranked,axis=-1,keepdims=True))
    with np.errstate(invalid='ignore',divide='ignore'):ranked=ranked/norm
    constant=norm[:,:,0]==0;ranked=np.nan_to_num(ranked,nan=0.);r=np.einsum('bpn,bqn->bpq',ranked,ranked,optimize=False)
    for j in range(len(columns)):r[:,j,:][constant[:,j]]=np.nan;r[:,:,j][constant[:,j]]=np.nan
    ids=g.patient_id.to_numpy(str);result=[];RF={}
    for target in ['EDV','SV','EF']:
        errors=g[target].to_numpy(float);oracle=aurc(errors,np.lexsort((ids,-errors)),idx,n);random=np.where(valid,errors[safe],0.).sum(axis=1)/n;den=random-oracle
        for score in NEW_SCORES+(['dice_oracle'] if has_dice else [])+['true_error_oracle']:
            values=g.dice_error.to_numpy(float) if score=='dice_oracle' else errors if score=='true_error_oracle' else g[score].to_numpy(float)
            detector=aurc(errors,np.lexsort((ids,-values)),idx,n)
            with np.errstate(invalid='ignore',divide='ignore'):rf=(random-detector)/den
            rf[np.abs(den)<=np.finfo(float).eps]=np.nan;RF[(target,score)]=rf
    for spec in STAT_SPECS:
        metric,target,score=spec
        if not has_dice and dice_dependent(spec):value=np.full(len(idx),np.nan)
        elif metric=='rho_score_target':value=r[:,lookup[score],lookup[target]]
        elif metric=='paired_gap':value=r[:,lookup[score],lookup['dice_error']]-r[:,lookup[score],lookup['EF']]
        elif metric=='partial_rho_score_EF_given_dice':value=partial(r,lookup[score],lookup['EF'],lookup['dice_error'])
        elif metric=='rho_dice_EF':value=r[:,lookup['dice_error'],lookup['EF']]
        else:value=RF[(target,score)]
        result.append(value)
    return np.column_stack(result)
