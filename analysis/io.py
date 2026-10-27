"""Private input contract, portable paths and resource limits; no bundled records."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from cardiac_functional_qc import _locked_review
from .protocol import SCORES

def low_priority():
    import psutil
    p=psutil.Process()
    if os.name=='nt':
        p.nice(psutil.IDLE_PRIORITY_CLASS);p.ionice(psutil.IOPRIO_VERYLOW)
    else:
        try:p.nice(19)
        except psutil.AccessDenied:pass
    if hasattr(p,'cpu_affinity'):
        try:p.cpu_affinity([p.cpu_affinity()[0]])
        except (psutil.AccessDenied,OSError):pass

def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--draws',type=int,default=5000)
    p.add_argument('--scores',nargs='+',default=SCORES)
    p.add_argument('--sensitivity',choices=['primary','exclude_pathological_10_rows','exclude_reference_3'],default='primary')
    return p

def boolean(values):
    return values.astype(str).str.lower().isin(['true','1','yes'])

def load(path,sensitivity='primary'):
    d=pd.read_csv(path,dtype={'patient_id':str},float_precision='round_trip')
    if 'pred_ef_points' in d:d=d.rename(columns={'pred_ef_points':'pred_ef','ref_ef_points':'ref_ef'})
    required=['cohort','patient_id','architecture','ventricle','pred_edv_ml','pred_esv_ml','ref_edv_ml','ref_esv_ml','pred_ef','ref_ef','dice']
    if missing:=set(required)-set(d):raise ValueError('Missing input columns: '+', '.join(sorted(missing)))
    if d.duplicated(['cohort','patient_id','architecture','ventricle']).any():raise ValueError('Duplicate patient-model rows')
    d['cohort']=d.cohort.replace({'M&Ms':'MNMS','M&Ms ':'MNMS','M&Ms2':'MNMS'})
    for who in ['pred','ref']:
        if who+'_sv_ml' not in d:d[who+'_sv_ml']=d[who+'_edv_ml']-d[who+'_esv_ml']
    for target,field in [('EDV','edv_ml'),('SV','sv_ml'),('EF','ef')]:
        calculated=abs(d['pred_'+field]-d['ref_'+field])
        if target in d:
            if not np.allclose(d[target],calculated,rtol=0,atol=1e-8,equal_nan=True):raise ValueError('Stored absolute error disagrees with measurement columns')
        else:d[target]=calculated
    d['dice_error']=1-d.dice
    d['relative_EDV']=d.EDV/d.ref_edv_ml
    d['relative_SV']=d.SV/d.ref_sv_ml
    d['relative_EDV_error']=d.relative_EDV
    if 'pathological_row' in d:d['pathological_row']=boolean(d.pathological_row)
    else:
        if sensitivity=='exclude_pathological_10_rows':raise ValueError('Supply the frozen private pathological_row flags; do not substitute a ventricular flag set')
        d['pathological_row']=False
    d=select_sensitivity(d,sensitivity)
    return d.sort_values(['cohort','ventricle','architecture','patient_id']).reset_index(drop=True)


def select_sensitivity(d,sensitivity):
    if sensitivity=='exclude_pathological_10_rows':d=d[~d.pathological_row]
    if sensitivity=='exclude_reference_3':
        if 'exclude_reference' not in d:raise ValueError('Supply a private exclude_reference flag; no reference IDs are stored in the source')
        d=d[~(d.cohort.eq('MNMS')&boolean(d.exclude_reference))]
    return d.copy()

def save_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')

def summary(rows,axes):
    d=pd.DataFrame(rows);result=[]
    for values,g in d.groupby(axes,dropna=False,sort=False):
        if not isinstance(values,tuple):values=(values,)
        a=g.estimate.to_numpy(float)
        result.append(dict(zip(axes,values),architecture='median_across_architectures',estimate=float(np.median(a)),architecture_min=float(np.min(a)),architecture_max=float(np.max(a)),architectures=len(a),n_min=int(g.n.min()),n_max=int(g.n.max())))
    return result

def write_ledger(path,rows):
    Path(path).write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
