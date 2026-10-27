"""Joint patient bootstrap of Eq.13 RF and RF(Dice oracle)-RF(score)."""
import hashlib
import numpy as np
from .io import parser,load,save_rows,write_ledger,low_priority
from .protocol import SCORES,seed_for
from .statistics import rf_occurrences,finite_interval

def compute(d,output,B=5000,scores=SCORES,sensitivity='primary'):
    output.mkdir(parents=True,exist_ok=True);rows=[];ledger=[]
    for (cohort,vent),part in d.groupby(['cohort','ventricle'],sort=True):
        models=sorted(part.architecture.unique());master=sorted(part.patient_id.unique())
        for target in ['EDV','SV','EF']:
            gs=[part[part.architecture.eq(a)].sort_values('patient_id').reset_index(drop=True) for a in models]
            maps=[np.array([{v:i for i,v in enumerate(g.patient_id)}.get(v,-1) for v in master]) for g in gs]
            matrices=[g[[*scores,'dice_error']].to_numpy(float) for g in gs];errors=[g[target].to_numpy(float) for g in gs]
            points=np.stack([rf_occurrences(x,e)[0] for x,e in zip(matrices,errors)])
            seed,cell=seed_for(sensitivity,cohort,vent,target);rng=np.random.default_rng(seed);digest=hashlib.sha256();draws=np.empty((B,len(models),len(scores)+1))
            for first in range(0,B,64):
                stop=min(B,first+64);sample=rng.integers(0,len(master),size=(stop-first,len(master)));digest.update(sample.tobytes())
                for offset,ix in enumerate(sample):
                    ix=ix[np.argsort(ix,kind='mergesort')]
                    for ai in range(len(models)):
                        local=maps[ai][ix];local=local[local>=0]
                        draws[first+offset,ai]=rf_occurrences(matrices[ai][local],errors[ai][local])[0]
            ledger.append(dict(cohort=cohort,ventricle=vent,target=target,seed=seed,cell=cell,draws=B,draw_index_sha256=digest.hexdigest()))
            for j,policy in enumerate([*scores,'dice_oracle']):
                for ai,arch in enumerate(models):
                    rows.append(dict(cohort=cohort,ventricle=vent,target=target,score=policy,architecture=arch,estimand='RF_per_architecture',n=len(gs[ai]),estimate=float(points[ai,j]),seed=seed,**(finite_interval(draws[:,ai,j]) if B else {})))
                rows.append(dict(cohort=cohort,ventricle=vent,target=target,score=policy,architecture='median_across_architectures',estimand='median_RF',n=len(master),estimate=float(np.median(points[:,j])),seed=seed,**(finite_interval(np.median(draws[:,:,j],axis=1)) if B else {})))
                if policy=='dice_oracle':continue
                differences=draws[:,:,-1]-draws[:,:,j];point_diff=points[:,-1]-points[:,j]
                for ai,arch in enumerate(models):
                    rows.append(dict(cohort=cohort,ventricle=vent,target=target,score=policy,architecture=arch,estimand='RF_Dice_minus_RF_QC_per_architecture',n=len(gs[ai]),estimate=float(point_diff[ai]),seed=seed,**(finite_interval(differences[:,ai]) if B else {})))
                rows.append(dict(cohort=cohort,ventricle=vent,target=target,score=policy,architecture='median_across_architectures',estimand='median_of_paired_RF_differences',n=len(master),estimate=float(np.median(point_diff)),seed=seed,**(finite_interval(np.median(differences,axis=1)) if B else {})))
    save_rows(output/'recovered_fraction_intervals.csv',rows);write_ledger(output/'seed_ledger.json',ledger)
    return rows

def main():
    a=parser(__doc__).parse_args();low_priority();compute(load(a.input,a.sensitivity),a.output,a.draws,a.scores,a.sensitivity)

if __name__=='__main__':main()
