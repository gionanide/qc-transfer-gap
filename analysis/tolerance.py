"""Table VI/S9: retained exceedance, robust EF error and required review fraction."""
import hashlib
import numpy as np
from .io import parser,load,save_rows,write_ledger,low_priority
from .protocol import SCORES,stable_seed
from .statistics import counts_from_master,describe,finite_interval,extended_interval
from . import policy_kernel as kernel

def compute(d,output,B=5000,scores=SCORES,has_dice=True):
    output.mkdir(parents=True,exist_ok=True)
    kernel.SCORES=list(scores)
    kernel.POLICIES=['random']+list(scores)+(['dice_oracle'] if has_dice else [])+['plausibility_random']+['plausibility_'+s for s in scores]+['true_error_oracle']
    rows=[];ledger=[];all_specs=kernel.specifications(False);boot_specs=kernel.specifications(True)
    for cohort,part in d.groupby('cohort',sort=True):
        seed=(20260907 if cohort=='ACDC' else 20260908) if has_dice else stable_seed('step5_four_model_mean',cohort)
        master=sorted(part.patient_id.unique());rng=np.random.default_rng(seed);indices=rng.integers(0,len(master),size=(B,len(master)),dtype=np.int32)
        ledger.append(dict(cohort=cohort,seed=seed,draws=B,draw_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
        saved={}
        for (vent,arch),g in part.groupby(['ventricle','architecture'],sort=True):
            g=g.sort_values('patient_id');ctx=kernel.make_context(g)
            if not has_dice:ctx['orders'].pop('dice_oracle',None)
            point=kernel.evaluate(ctx,np.ones((1,len(g)),np.int16),False)
            lookup={v:i for i,v in enumerate(g.patient_id)};mapping=np.array([lookup.get(v,-1) for v in master])
            draws=np.empty((B,len(boot_specs)))
            for first in range(0,B,64):
                stop=min(B,first+64);counts=counts_from_master(indices[first:stop],mapping,len(g));values=kernel.evaluate(ctx,counts,True)
                draws[first:stop]=np.column_stack([values[s] for s in boot_specs])
            for spec in all_specs:
                r=dict(cohort=cohort,ventricle=vent,architecture=arch,n=len(g),seed=seed,estimate=float(point[spec][0]),**describe(spec))
                if spec in boot_specs and B:r.update((extended_interval if spec[0]=='required_budget' else finite_interval)(draws[:,boot_specs.index(spec)]))
                rows.append(r)
            saved.setdefault(vent,[]).append((point,draws,len(g)))
        for vent,cells in saved.items():
            for spec in all_specs:
                r=dict(cohort=cohort,ventricle=vent,architecture='median_across_architectures',n=len(master),seed=seed,estimate=float(np.median([x[0][spec][0] for x in cells])),**describe(spec))
                if spec in boot_specs and B:
                    values=np.median(np.column_stack([x[1][:,boot_specs.index(spec)] for x in cells]),axis=1)
                    r.update((extended_interval if spec[0]=='required_budget' else finite_interval)(values))
                rows.append(r)
    for kind in ['exceedance','robust','required_budget']:save_rows(output/(kind+'.csv'),[r for r in rows if r['analysis']==kind])
    write_ledger(output/'seed_ledger.json',ledger)
    return rows

def main():
    a=parser(__doc__).parse_args();low_priority();compute(load(a.input,a.sensitivity),a.output,a.draws,a.scores)

if __name__=='__main__':main()
