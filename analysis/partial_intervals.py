"""Partial-Spearman CIs: prior per-model seeds and joint architecture-median draws."""
import hashlib
import numpy as np
from .io import parser,load,save_rows,write_ledger,low_priority
from .protocol import gap_seed,seed_for
from .statistics import rank_correlations,partial,finite_interval

def main():
    a=parser(__doc__).parse_args();low_priority();d=load(a.input,a.sensitivity);rows=[];ledger=[]
    for (cohort,vent),part in d.groupby(['cohort','ventricle'],sort=True):
        models=sorted(part.architecture.unique());groups=[part[part.architecture.eq(m)].sort_values('patient_id') for m in models]
        ids=sorted(part.patient_id.unique())
        for target in ['EDV','SV','EF']:
            joint_seed,_=seed_for(a.sensitivity,cohort,vent,target)
            for score in a.scores:
                points=[];joint=np.empty((a.draws,len(models)))
                for ai,(arch,g) in enumerate(zip(models,groups)):
                    x=g[[score,'dice_error',target]].to_numpy(float).T
                    point=float(partial(rank_correlations(x[None]),0,2,1)[0]);points.append(point)
                    seed=gap_seed(vent,cohort,arch,score);rng=np.random.default_rng(seed);draws=np.empty(a.draws);digest=hashlib.sha256()
                    for first in range(0,a.draws,64):
                        stop=min(a.draws,first+64);idx=rng.integers(0,len(g),size=(stop-first,len(g)));digest.update(idx.tobytes())
                        draws[first:stop]=partial(rank_correlations(np.transpose(x[:,idx],(1,0,2))),0,2,1)
                    rows.append(dict(cohort=cohort,ventricle=vent,target=target,score=score,architecture=arch,n=len(g),estimate=point,seed=seed,**(finite_interval(draws) if a.draws else {})))
                    ledger.append(dict(cohort=cohort,ventricle=vent,target=target,score=score,architecture=arch,seed=seed,draw_index_sha256=digest.hexdigest()))
                    rng=np.random.default_rng(joint_seed);lookup={v:i for i,v in enumerate(g.patient_id)};mapping=np.array([lookup.get(v,-1) for v in ids])
                    for first in range(0,a.draws,64):
                        stop=min(a.draws,first+64);idx=mapping[rng.integers(0,len(ids),size=(stop-first,len(ids)))]
                        if np.all(idx>=0):joint[first:stop,ai]=partial(rank_correlations(np.transpose(x[:,idx],(1,0,2))),0,2,1)
                        else:
                            for offset,local in enumerate(idx):
                                local=local[local>=0]
                                joint[first+offset,ai]=partial(rank_correlations(x[:,local][None]),0,2,1)[0] if len(local)>=3 else np.nan
                samples=np.median(joint,axis=1)
                rows.append(dict(cohort=cohort,ventricle=vent,target=target,score=score,architecture='median_across_architectures',n=len(ids),estimate=float(np.median(points)),seed=joint_seed,**(finite_interval(samples) if a.draws else {})))
    a.output.mkdir(parents=True,exist_ok=True);save_rows(a.output/'partial_correlation_intervals.csv',rows);write_ledger(a.output/'seed_ledger.json',ledger)

if __name__=='__main__':main()
