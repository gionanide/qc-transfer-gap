"""Table S8: official M&Ms vendors, including Siemens versus non-Siemens."""
import hashlib
import numpy as np
import pandas as pd
from .io import parser,load,save_rows,write_ledger,low_priority
from .protocol import stable_seed
from .statistics import finite_interval
from .vendor_kernel import statistics,SPECS

def compute(d,output,B=5000):
    output.mkdir(parents=True,exist_ok=True);d=d[d.cohort.eq('MNMS')].copy()
    if 'vendor' not in d:raise ValueError('Join official vendor codes into the private input, or supply --metadata')
    rows=[];ledger=[];saved={}
    for group in ['A','B','C','D','non_Siemens']:
        part=d[d.vendor.ne('A')] if group=='non_Siemens' else d[d.vendor.eq(group)]
        master=sorted(part.patient_id.unique());seed=stable_seed('step4_vendor','MNMS',group);rng=np.random.default_rng(seed);indices=rng.integers(0,len(master),size=(B,len(master)))
        ledger.append(dict(subgroup=group,seed=seed,n=len(master),draws=B,draw_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
        cells={}
        for (vent,arch),g in part.groupby(['ventricle','architecture'],sort=True):
            g=g.sort_values('patient_id');lookup={v:i for i,v in enumerate(g.patient_id)};mapping=np.array([lookup.get(v,-1) for v in master])
            point=statistics(g,mapping[None])[0];boot=np.empty((B,len(SPECS)))
            for first in range(0,B,64):
                stop=min(B,first+64);boot[first:stop]=statistics(g,mapping[indices[first:stop]])
            cells[vent,arch]=(point,boot,len(g))
            for j,(metric,target,score) in enumerate(SPECS):rows.append(dict(subgroup=group,ventricle=vent,architecture=arch,metric=metric,target=target,score=score,n=len(g),estimate=float(point[j]),**(finite_interval(boot[:,j]) if B else {})))
        for vent in sorted(part.ventricle.unique()):
            selected=[x for (v,a),x in cells.items() if v==vent]
            for j,(metric,target,score) in enumerate(SPECS):
                point=np.array([x[0][j] for x in selected]);samples=np.median(np.column_stack([x[1][:,j] for x in selected]),axis=1)
                rows.append(dict(subgroup=group,ventricle=vent,architecture='median_across_architectures',metric=metric,target=target,score=score,n=len(master),estimate=float(np.median(point)),architecture_min=float(point.min()),architecture_max=float(point.max()),**(finite_interval(samples) if B else {})))
        if group in ['A','non_Siemens']:saved[group]=cells
    contrasts=[]
    for vent,arch in saved['A']:
        a,aa,na=saved['A'][vent,arch];b,bb,nb=saved['non_Siemens'][vent,arch]
        for j,(metric,target,score) in enumerate(SPECS):
            if metric in ['paired_gap','EF_recovered_fraction']:contrasts.append(dict(ventricle=vent,architecture=arch,metric=metric,target=target,score=score,estimate=float(a[j]-b[j]),**(finite_interval(aa[:,j]-bb[:,j]) if B else {})))
    save_rows(output/'vendor_results.csv',rows);save_rows(output/'Siemens_minus_non_Siemens.csv',contrasts);write_ledger(output/'seed_ledger.json',ledger)
    return rows

def main():
    p=parser(__doc__);p.add_argument('--metadata',type=str);a=p.parse_args();low_priority();d=load(a.input,a.sensitivity)
    if a.metadata:
        meta=pd.read_csv(a.metadata,dtype={'External code':str});meta=meta.rename(columns={'External code':'patient_id','Vendor':'vendor'})
        if 'VendorName' in meta and 'vendor' not in meta:raise ValueError('Use the official vendor-code column, not manufacturer names')
        mapping=meta[['patient_id','vendor']].drop_duplicates();d=d.drop(columns=['vendor'],errors='ignore').merge(mapping,on='patient_id',how='left',validate='many_to_one')
    compute(d,a.output,a.draws)

if __name__=='__main__':main()
