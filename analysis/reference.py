"""Reference/Dice-oracle/partial/relative analyses of user-supplied patient outputs."""
import hashlib
import numpy as np
import pandas as pd
from .io import parser,load,save_rows,summary,write_ledger,low_priority
from .protocol import SCORES,corr_seed,gap_seed,stable_seed
from .statistics import rank_correlations,partial,finite_interval
from .risk_coverage import evaluate_deferral

def ranked_stat(g,columns,fn,seed,B):
    x=g[columns].to_numpy(float).T
    point=float(fn(rank_correlations(x[None]))[0])
    draws=np.empty(B);rng=np.random.default_rng(seed);digest=hashlib.sha256()
    for first in range(0,B,64):
        stop=min(B,first+64);ix=rng.integers(0,len(g),size=(stop-first,len(g)));digest.update(ix.tobytes())
        draws[first:stop]=fn(rank_correlations(np.transpose(x[:,ix],(1,0,2))))
    result=dict(estimate=point,seed=int(seed))
    if B:result.update(finite_interval(draws))
    return result,digest.hexdigest()

def compute(d,output,B=5000,scores=SCORES,relative=False):
    output.mkdir(parents=True,exist_ok=True);rows=[];ledger=[];curves=[]
    targets=['relative_EDV','relative_SV'] if relative else ['EDV','SV','EF']
    for (cohort,vent,arch),original in d.groupby(['cohort','ventricle','architecture'],sort=True):
        if not relative:
            for score in scores:
                g=original.sort_values('patient_id');g=g[np.isfinite(g[score])&np.isfinite(g.dice_error)]
                if len(g)<3:continue
                ctx=dict(cohort=cohort,ventricle=vent,architecture=arch,target='dice_error',score=score,n=len(g))
                seed=corr_seed(vent,cohort,arch,score,'dice');result,h=ranked_stat(g,[score,'dice_error'],lambda r:r[:,0,1],seed,B)
                rows.append(dict(ctx,metric='rho_score_dice_error',**result));ledger.append(dict(ctx,metric='rho_score_dice_error',seed=seed,draw_index_sha256=h))
        for target in targets:
            g=original.sort_values('patient_id')
            if target=='relative_EDV':g=g[g.ref_edv_ml>0]
            if target=='relative_SV':g=g[g.ref_sv_ml>0]
            g=g[np.isfinite(g[target])&np.isfinite(g.dice_error)].copy()
            context=dict(cohort=cohort,ventricle=vent,architecture=arch,target=target,n=len(g))
            if len(g)<3:continue
            for score in scores:
                work=g[np.isfinite(g[score])].copy()
                if len(work)<3:continue
                ctx=dict(context,score=score,n=len(work))
                native=target.removeprefix('relative_')
                seed=corr_seed(vent,cohort,arch,score,native) if not relative else stable_seed('relative_error',vent,cohort,arch,score,target)
                result,h=ranked_stat(work,[score,target],lambda r:r[:,0,1],seed,B)
                rows.append(dict(ctx,metric='rho_score_error',**result));ledger.append(dict(ctx,metric='rho_score_error',seed=seed,draw_index_sha256=h))
                seed=gap_seed(vent,cohort,arch,score)
                for metric,fn in [('partial_spearman_given_dice_error',lambda r:partial(r,0,2,1)),('paired_gap',lambda r:r[:,0,1]-r[:,0,2])]:
                    result,h=ranked_stat(work,[score,'dice_error',target],fn,seed,B)
                    rows.append(dict(ctx,metric=metric,**result));ledger.append(dict(ctx,metric=metric,seed=seed,draw_index_sha256=h))
                result=evaluate_deferral(work[score],work[target]);rows.append(dict(ctx,metric='recovered_fraction',estimate=float(result['normalized_deferral_gain'])))
                for coverage,risk in zip(result['coverage'],result['detector_risk']):curves.append(dict(ctx,coverage=float(coverage),retained_mean_absolute_error=float(risk)))
            for policy,values in [('dice_oracle',g.dice_error),('true_error_oracle',g[target])]:
                result=evaluate_deferral(values,g[target]);rows.append(dict(context,score=policy,metric='recovered_fraction',estimate=float(result['normalized_deferral_gain'])))
                for coverage,risk in zip(result['coverage'],result['detector_risk']):curves.append(dict(context,score=policy,coverage=float(coverage),retained_mean_absolute_error=float(risk)))
            result,h=ranked_stat(g,['dice_error',target],lambda r:r[:,0,1],stable_seed('dice_reference',vent,cohort,arch,target),B)
            rows.append(dict(context,score='dice_oracle',metric='rho_dice_error_measurement_error',**result))
    axes=['cohort','ventricle','target','score','metric']
    summaries=summary(rows,axes)
    save_rows(output/'per_architecture.csv',rows);save_rows(output/'architecture_summaries.csv',summaries);save_rows(output/'risk_coverage.csv',curves)
    write_ledger(output/'seed_ledger.json',ledger)
    partials=pd.DataFrame([r for r in summaries if r['metric']=='partial_spearman_given_dice_error'])
    s6=[]
    for (cohort,vent,target),g in partials.groupby(['cohort','ventricle','target'],sort=False):
        rf=next(r for r in summaries if r['cohort']==cohort and r['ventricle']==vent and r['target']==target and r['metric']=='recovered_fraction' and r['score']=='dice_oracle')
        r=dict(cohort=cohort,ventricle=vent,target=target,dice_oracle_RF_median=rf['estimate'],dice_oracle_RF_min=rf['architecture_min'],dice_oracle_RF_max=rf['architecture_max'])
        r.update({'partial_'+x.score:x.estimate for x in g.itertuples()});r['partial_min_across_scores']=float(g.estimate.min());r['partial_max_across_scores']=float(g.estimate.max());s6.append(r)
    save_rows(output/'S6_reference_summary.csv',s6)
    return rows

def main():
    p=parser(__doc__);p.add_argument('--relative',action='store_true');a=p.parse_args();low_priority()
    compute(load(a.input,a.sensitivity),a.output,a.draws,a.scores,a.relative)

if __name__=='__main__':main()
