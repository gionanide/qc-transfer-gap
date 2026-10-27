"""Architecture or fold SD scores, distinct from numerical-mean prediction errors."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .io import load,save_rows,low_priority,select_sensitivity
from .protocol import ARCHS
from .spread_evaluation import compute
from .tolerance import compute as tolerance

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--draws',type=int,default=5000);p.add_argument('--fold-input');p.add_argument('--sensitivity',default='primary',choices=['primary','exclude_pathological_10_rows','exclude_reference_3']);a=p.parse_args();low_priority();d=load(a.input,a.sensitivity)
    original=load(a.input)
    prefix='fold' if a.fold_input else 'architecture'
    names=[prefix+'_EF_SD',prefix+'_EDV_SD',prefix+'_SV_SD'];rows=[];means=[]
    source=pd.read_csv(a.fold_input,dtype={'patient_id':str},float_precision='round_trip') if a.fold_input else original[original.architecture.isin(ARCHS)]
    if a.fold_input and 'architecture' not in source:source['architecture']=source['fold'].astype(str)
    if 'pred_sv_ml' not in source:source['pred_sv_ml']=source.pred_edv_ml-source.pred_esv_ml
    for values,g in source.groupby(['cohort','ventricle','patient_id'],sort=True):
        if not a.fold_input and set(g.architecture)!=set(ARCHS):raise ValueError('Four fixed architectures are required')
        row=dict(zip(['cohort','ventricle','patient_id'],values))
        for target,column in [('EF','pred_ef'),('EDV','pred_edv_ml'),('SV','pred_sv_ml')]:row[prefix+'_'+target+'_SD']=float(np.std(g[column].to_numpy(),ddof=0))
        rows.append(row)
        if not a.fold_input:
            mean=g.iloc[0].to_dict();mean['architecture']='four_model_mean';mean['pathological_row']=bool(g.pathological_row.any())
            for column in ['pred_ef','pred_edv_ml','pred_esv_ml','pred_sv_ml']:mean[column]=float(g[column].mean())
            for target,column in [('EF','ef'),('EDV','edv_ml'),('SV','sv_ml')]:mean[target]=abs(mean['pred_'+column]-mean['ref_'+column])
            mean['dice']=np.nan;mean['dice_error']=np.nan;mean.update(row);means.append(mean)
    scored=d.merge(pd.DataFrame(rows),on=['cohort','ventricle','patient_id'],validate='many_to_one')
    save_rows(a.output/'derived_scores.csv',scored.to_dict('records'))
    compute(scored,a.output/'single_architecture',a.draws,names);tolerance(scored,a.output/'clinical',a.draws,names)
    if not a.fold_input:
        mean_table=select_sensitivity(pd.DataFrame(means),a.sensitivity)
        save_rows(a.output/'numerical_mean_predictions.csv',mean_table.to_dict('records'))
        save_rows(a.output/'numerical_mean_Dice_status.csv',[dict(estimand='Dice-dependent numerical-mean analyses',status='undefined_no_mean_prediction_mask_or_Dice')])
        mean_table['relative_EDV_error']=mean_table.EDV/mean_table.ref_edv_ml
        compute(mean_table,a.output/'four_model_mean',a.draws,names,False)
        tolerance(mean_table,a.output/'four_model_mean_clinical',a.draws,names,False)

if __name__=='__main__':main()
