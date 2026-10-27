"""Joint patient resampling for spread scores and paired score comparisons."""
import hashlib
import numpy as np
from . import spread_statistics as kernel
from .protocol import stable_seed
from .statistics import finite_interval
from .io import save_rows,write_ledger

def compute(d,output,B,scores,has_dice=True):
    output.mkdir(parents=True,exist_ok=True);kernel.NEW_SCORES=list(scores)
    specs=[('rho_score_target',t,s) for s in scores for t in ['dice_error','EDV','SV','EF']]
    specs += [('paired_gap','EF',s) for s in scores]+[('partial_rho_score_EF_given_dice','EF',s) for s in scores]
    specs += [('recovered_fraction',t,s) for t in ['EDV','SV','EF'] for s in [*scores,'dice_oracle','true_error_oracle']]
    specs += [('rho_dice_EF','EF','dice_oracle')];kernel.STAT_SPECS=specs
    extra=[]
    if 'MC_EF_SD' in scores:
        extra.append(('RF_Dice_minus_EF_SD','EF','MC_EF_SD'))
        for s in ['new_mc_dropout','new_measurement_aware_mc_dropout']:
            if s in scores:extra.extend([('rho_EF_SD_minus_new_entropy','EF',s),('RF_EF_SD_minus_new_entropy','EF',s)])
    def values(g,idx):
        base=kernel.stat_values(g,idx,has_dice);added=[]
        for metric,target,score in extra:
            if metric=='RF_Dice_minus_EF_SD':left=('recovered_fraction','EF','dice_oracle');right=('recovered_fraction','EF','MC_EF_SD')
            else:
                name='rho_score_target' if metric.startswith('rho_') else 'recovered_fraction';left=(name,'EF','MC_EF_SD');right=(name,'EF',score)
            added.append(base[:,specs.index(left)]-base[:,specs.index(right)])
        return np.column_stack([base,*added]) if added else base
    fullspecs=specs+extra;rows=[];ledger=[]
    for cohort,original in d.groupby('cohort',sort=True):
        groups=['all']+(['Siemens','non_Siemens'] if cohort=='MNMS' and 'vendor' in d else [])
        for group in groups:
            part=original if group=='all' else original[original.vendor.eq('A') if group=='Siemens' else original.vendor.ne('A')]
            seed=(20260907 if cohort=='ACDC' else 20260908) if has_dice else stable_seed('step5_four_model_mean',cohort)
            if group!='all':seed=stable_seed('step5B_vendor',group)
            master=sorted(part.patient_id.unique());rng=np.random.default_rng(seed);indices=rng.integers(0,len(master),size=(B,len(master)),dtype=np.int32)
            ledger.append(dict(cohort=cohort,vendor_group=group,seed=seed,draws=B,index_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
            saved={}
            for (vent,arch),g in part.groupby(['ventricle','architecture'],sort=True):
                g=g.sort_values('patient_id');lookup={v:i for i,v in enumerate(g.patient_id)};mapping=np.array([lookup.get(v,-1) for v in master])
                point=values(g,mapping[None])[0];boot=np.empty((B,len(fullspecs)))
                for first in range(0,B,64):
                    stop=min(B,first+64);boot[first:stop]=values(g,mapping[indices[first:stop]])
                saved.setdefault(vent,[]).append((point,boot))
                for j,(metric,target,score) in enumerate(fullspecs):rows.append(dict(cohort=cohort,vendor_group=group,ventricle=vent,architecture=arch,metric=metric,target=target,score=score,n=len(g),estimate=float(point[j]),seed=seed,**(finite_interval(boot[:,j]) if B else {})))
            for vent,cells in saved.items():
                for j,(metric,target,score) in enumerate(fullspecs):
                    point=np.array([x[0][j] for x in cells]);samples=np.median(np.column_stack([x[1][:,j] for x in cells]),axis=1)
                    rows.append(dict(cohort=cohort,vendor_group=group,ventricle=vent,architecture='median_across_architectures',metric=metric,target=target,score=score,n=len(master),estimate=float(np.median(point)),architecture_min=float(point.min()),architecture_max=float(point.max()),seed=seed,**(finite_interval(samples) if B else {})))
    save_rows(output/'spread_evaluation.csv',rows);write_ledger(output/'seed_ledger.json',ledger)
    return rows
