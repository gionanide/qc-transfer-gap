"""Measurement-space MC scores; errors remain those of the deterministic deployment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .io import load,low_priority,save_rows
from .spread_evaluation import compute
from .tolerance import compute as tolerance

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--passes',required=True);p.add_argument('--deterministic-input',required=True);p.add_argument('--entropy-input');p.add_argument('--output',required=True,type=Path);p.add_argument('--draws',type=int,default=5000);p.add_argument('--sensitivity',default='primary',choices=['primary','exclude_pathological_10_rows','exclude_reference_3']);a=p.parse_args();low_priority()
    d=pd.read_csv(a.passes,dtype={'patient_id':str},float_precision='round_trip')
    fields=['cohort','architecture','patient_id','ventricle','pass_ordinal']
    pairs=d.pivot(index=fields,columns='phase',values='native_volume_ml').reset_index().rename(columns={'ED':'EDV','ES':'ESV'})
    pairs['SV']=pairs.EDV-pairs.ESV;pairs['EF']=np.where(pairs.EDV>0,100*pairs.SV/pairs.EDV,np.nan)
    rows=[]
    for values,g in pairs.groupby(fields[:-1],sort=True):
        if len(g)!=20:raise ValueError('Exactly twenty paired ED/ES passes are required')
        row=dict(zip(fields[:-1],values))
        for target in ['EF','EDV','SV']:row['MC_'+target+'_SD']=float(np.std(g[target].to_numpy(),ddof=0))
        q=np.quantile(g.EF,[.25,.75],method='linear');row['MC_EF_IQR']=float(q[1]-q[0]);rows.append(row)
    scores=['MC_EF_SD','MC_EDV_SD','MC_SV_SD','MC_EF_IQR']
    original=load(a.deterministic_input,a.sensitivity);scored=original.merge(pd.DataFrame(rows),on=fields[:-1],validate='one_to_one')
    if a.entropy_input:
        ent=pd.read_csv(a.entropy_input,dtype={'patient_id':str},float_precision='round_trip')
        cols=['new_mc_dropout','new_measurement_aware_mc_dropout']
        scored=scored.drop(columns=cols,errors='ignore').merge(ent[fields[:-1]+cols],on=fields[:-1],validate='one_to_one')
    # The recomputed MC entropy must be supplied from the same new passes for direct comparisons.
    if 'new_mc_dropout' in scored:scores+=['new_mc_dropout','new_measurement_aware_mc_dropout']
    save_rows(a.output/'derived_scores.csv',scored.to_dict('records'))
    compute(scored,a.output/'association',a.draws,scores);tolerance(scored,a.output/'clinical',a.draws,scores)

if __name__=='__main__':main()
