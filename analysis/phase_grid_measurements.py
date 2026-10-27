"""Table S7 phase measurements from private saved masks; no inference."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from training.geometry import _resample_inplane,_crop_or_pad,preprocess_volume,processed_prediction_to_native,dice_binary
from .io import save_rows,low_priority

def mask_array(path):
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:return z['pred_processed'].astype(np.int16)
    if path.suffix=='.npy':return np.load(path,allow_pickle=False).astype(np.int16)
    return np.asarray(nib.load(path).dataobj,dtype=np.int16)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args();low_priority();rows=[]
    for r in pd.read_csv(a.manifest,dtype={'patient_id':str}).itertuples(index=False):
        nii=nib.load(r.image);image=np.asarray(nii.dataobj,dtype=np.float32);ref=np.asarray(nib.load(r.reference).dataobj,dtype=np.int16)
        if image.ndim==4:image=image[...,int(r.frame)]
        if ref.ndim==4:ref=ref[...,int(r.frame)]
        if r.cohort=='MNMS':ref=np.take(np.array([0,3,2,1],np.int16),ref)
        spacing=nii.header.get_zooms()[:3];_,processed_ref,shapes=preprocess_volume(image,ref,*map(float,spacing[:2]))
        if getattr(r,'native_prediction',None) and isinstance(r.native_prediction,str):
            native=mask_array(r.native_prediction)
            pro=np.stack([_crop_or_pad(_resample_inplane(native[:,:,z],spacing[:2],(1.25,1.25),True),(256,256),0) for z in range(native.shape[2])])
        else:
            pro=mask_array(r.processed_prediction);native=processed_prediction_to_native(pro,shapes,image.shape)
        for vent,label in [('RV',1),('LV',3)]:
            rows.append(dict(cohort=r.cohort,patient_id=r.patient_id,architecture=r.architecture,ventricle=vent,phase=r.phase,pred_native_ml=float((native==label).sum()*np.prod(spacing)/1000),ref_native_ml=float((ref==label).sum()*np.prod(spacing)/1000),pred_processed_ml=float((pro==label).sum()*1.25*1.25*spacing[2]/1000),ref_processed_ml=float((processed_ref==label).sum()*1.25*1.25*spacing[2]/1000),dice_processed=dice_binary(pro==label,processed_ref==label),dice_native=dice_binary(native==label,ref==label)))
    save_rows(a.output,rows)

if __name__=='__main__':main()
