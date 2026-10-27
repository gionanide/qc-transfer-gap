"""Table S7: processed/native endpoints from saved phase-grid measurements."""
import numpy as np
import pandas as pd
from .io import parser,load,save_rows,low_priority
from .reference import compute

def main():
    p=parser(__doc__);p.add_argument('--phase-table',required=True);a=p.parse_args();low_priority();d=load(a.input,a.sensitivity)
    phases=pd.read_csv(a.phase_table,dtype={'patient_id':str},float_precision='round_trip')
    records=[]
    for row in d.itertuples(index=False):
        phase=phases[phases.cohort.eq(row.cohort)&phases.architecture.eq(row.architecture)&phases.patient_id.eq(row.patient_id)&phases.ventricle.eq(row.ventricle)].set_index('phase')
        if set(phase.index)!=set(['ED','ES']):raise ValueError('Every selected row needs paired phase-grid measurements')
        records.append(dict(row._asdict(),grid='submitted'))
        for grid in ['processed','native']:
            r=dict(row._asdict(),grid=grid)
            for who in ['pred','ref']:
                ed=float(phase.loc['ED',who+'_'+grid+'_ml']);es=float(phase.loc['ES',who+'_'+grid+'_ml'])
                r.update({who+'_edv_ml':ed,who+'_esv_ml':es,who+'_sv_ml':ed-es,who+'_ef':100*(ed-es)/ed if ed>0 else np.nan})
            r['dice']=float((phase.loc['ED','dice_'+grid]+phase.loc['ES','dice_'+grid])/2)
            if grid=='processed':
                if abs(r['dice']-row.dice)>1e-6:raise ValueError('Recomputed processed Dice differs from submitted Dice')
                r['dice']=row.dice
            r['dice_error']=1-r['dice']
            for target,field in [('EDV','edv_ml'),('SV','sv_ml'),('EF','ef')]:r[target]=abs(r['pred_'+field]-r['ref_'+field])
            if grid=='native':
                for target in ['EDV','SV','EF']:
                    if abs(r[target]-getattr(row,target))>2e-6:raise ValueError('Reconstructed native endpoint differs from stored value')
                    r[target]=getattr(row,target)
            records.append(r)
    endpoints=pd.DataFrame(records);allrows=[]
    for grid,g in endpoints.groupby('grid',sort=False):
        rows=compute(g,a.output/grid,a.draws,a.scores)
        allrows.extend([dict(r,grid=grid) for r in rows])
    result=pd.DataFrame(allrows);base=result[result.grid.eq('submitted')][['cohort','ventricle','architecture','target','score','metric','estimate']].rename(columns={'estimate':'submitted_estimate'})
    result=result.merge(base,on=['cohort','ventricle','architecture','target','score','metric'],validate='many_to_one');result['difference_from_submitted']=result.estimate-result.submitted_estimate
    save_rows(a.output/'S7_grid_comparison.csv',result.to_dict('records'))

if __name__=='__main__':main()
