"""Convert the original saved RV/LV patient/QC tables into the private long contract."""
import argparse
from pathlib import Path
import pandas as pd
from .io import save_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acdc-patients','acdc-qc','mnms-patients','mnms-qc','lv-patients','lv-qc']:p.add_argument('--'+name,required=True)
    p.add_argument('--pathological-rows');p.add_argument('--output',required=True,type=Path);a=p.parse_args();frames=[]
    scores=['softmax_entropy','measurement_aware_softmax_entropy','mc_dropout','measurement_aware_mc_dropout']
    for cohort,patientfile,qcfile in [('ACDC',a.acdc_patients,a.acdc_qc),('MNMS',a.mnms_patients,a.mnms_qc)]:
        d=pd.read_csv(patientfile,dtype={'patient_id':str});q=pd.read_csv(qcfile,dtype={'patient_id':str})
        d=d.drop(columns=[s for s in scores if s in d]).merge(q[['architecture','patient_id',*scores]],on=['architecture','patient_id'],validate='one_to_one')
        d=d.rename(columns={'ref_rvef_points':'ref_ef','pred_rvef_points':'pred_ef','rv_dice_mean':'dice','abs_delta_edv_ml':'EDV','abs_delta_sv_ml':'SV','abs_delta_rvef_points':'EF'});d['cohort']=cohort;d['ventricle']='RV';frames.append(d)
    d=pd.read_csv(a.lv_patients,dtype={'patient_id':str}).merge(pd.read_csv(a.lv_qc,dtype={'patient_id':str}),on=['cohort','architecture','patient_id'],validate='one_to_one')
    d=d.rename(columns={'ref_lvedv_ml':'ref_edv_ml','ref_lvesv_ml':'ref_esv_ml','pred_lvedv_ml':'pred_edv_ml','pred_lvesv_ml':'pred_esv_ml','ref_lvsv_ml':'ref_sv_ml','pred_lvsv_ml':'pred_sv_ml','ref_lvef_points':'ref_ef','pred_lvef_points':'pred_ef','lv_dice_mean':'dice','abs_delta_lvedv_ml':'EDV','abs_delta_lvsv_ml':'SV','abs_delta_lvef_points':'EF'});d['ventricle']='LV';frames.append(d)
    fields=['cohort','patient_id','architecture','ventricle','pred_edv_ml','pred_esv_ml','ref_edv_ml','ref_esv_ml','pred_sv_ml','ref_sv_ml','pred_ef','ref_ef','dice','EDV','SV','EF',*scores]
    d=pd.concat([x[fields] for x in frames],ignore_index=True)
    if a.pathological_rows:
        flags=pd.read_csv(a.pathological_rows,dtype={'patient_id':str})[['cohort','architecture','patient_id']].drop_duplicates();flags['pathological_row']=True
        d=d.merge(flags,on=['cohort','architecture','patient_id'],how='left',validate='many_to_one');d['pathological_row']=d.pathological_row.fillna(False)
    save_rows(a.output,d.sort_values(['cohort','ventricle','architecture','patient_id']).to_dict('records'))

if __name__=='__main__':main()
