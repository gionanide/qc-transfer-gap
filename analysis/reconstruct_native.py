"""Native reconstruction and pure reference round trip, with the study inverse."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from training.geometry import preprocess_volume,processed_prediction_to_native,dice_binary
from .io import low_priority

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-image',required=True);p.add_argument('--processed-mask');p.add_argument('--reference');p.add_argument('--reference-labels',choices=['acdc','mnms'],default='acdc');p.add_argument('--frame',type=int);p.add_argument('--output',required=True,type=Path);a=p.parse_args();low_priority()
    nii=nib.load(a.native_image);image=np.asarray(nii.dataobj,dtype=np.float32)
    if image.ndim==4:
        if a.frame is None:raise ValueError('A frame index is required for 4D input')
        image=image[...,a.frame]
    sx,sy,sz=nii.header.get_zooms()[:3];ref=np.zeros(image.shape,np.int16)
    if a.reference:
        ref=np.asarray(nib.load(a.reference).dataobj)
        if ref.ndim==4:ref=ref[...,a.frame]
        if a.reference_labels=='mnms':ref=np.take(np.array([0,3,2,1],np.int16),ref.astype(np.int16))
    _,processed,shapes=preprocess_volume(image,ref,float(sx),float(sy))
    if a.processed_mask:
        if Path(a.processed_mask).suffix=='.npy':mask=np.load(a.processed_mask,allow_pickle=False)
        elif Path(a.processed_mask).suffix=='.npz':
            with np.load(a.processed_mask,allow_pickle=False) as z:mask=z['pred_processed']
        else:mask=np.asarray(nib.load(a.processed_mask).dataobj)
        if mask.shape==processed.transpose(1,2,0).shape:mask=mask.transpose(2,0,1)
    else:mask=processed
    native=processed_prediction_to_native(mask,shapes,image.shape)
    a.output.parent.mkdir(parents=True,exist_ok=True);nib.save(nib.Nifti1Image(native,nii.affine),a.output)
    if a.reference:
        for label,name in [(1,'RV'),(3,'LV')]:
            raw=(ref==label).sum()*sx*sy*sz/1000
            pro=(processed==label).sum()*1.25*1.25*sz/1000
            restored=(native==label).sum()*sx*sy*sz/1000
            print(name,'native_ml',raw,'processed_ml',pro,'roundtrip_ml',restored,'Dice',dice_binary(native==label,ref==label))

if __name__=='__main__':main()
