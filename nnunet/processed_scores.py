"""nnU-Net native ensemble probabilities to the paper grid for Dice and entropy.

Native volumes use the official nnU-Net exported mask. The processed mask
for Dice is nearest-neighbour resampled from that native hard mask. Each
probability channel is separately linearly resampled, then renormalised.
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from training.geometry import _resample_inplane,_crop_or_pad,dice_binary
from training.score_functions import struct_weight,agg

def processed(a,spacing,is_mask):
    return np.stack([_crop_or_pad(_resample_inplane(a[:,:,z],spacing,(1.25,1.25),is_mask),(256,256),0) for z in range(a.shape[2])],axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-mask',required=True);p.add_argument('--probabilities',required=True);p.add_argument('--reference');p.add_argument('--reference-labels',choices=['acdc','mnms'],default='acdc');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    nii=nib.load(a.native_mask);mask=np.asarray(nii.dataobj,dtype=np.uint8);spacing=nii.header.get_zooms();pro=processed(mask,spacing[:2],True)
    with np.load(a.probabilities,allow_pickle=False) as z:prob=z['probabilities']
    # SimpleITKIO/nnU-Net probability order is classes,z,y,x; NIfTI is x,y,z.
    if prob.shape[1:]!=mask.shape[::-1]:raise ValueError('Unexpected official native probability geometry')
    prob=prob.transpose(0,3,2,1)
    pp=np.stack([processed(channel,spacing[:2],False) for channel in prob],axis=0)
    total=pp.sum(axis=0);pp[0,total==0]=1.;total=pp.sum(axis=0);pp/=total[None]
    entropy=-np.sum(pp*np.log(np.maximum(pp,np.float32(1e-8))),axis=0)
    entropy=entropy[:,::2,::2][:,:128,:128]
    rows=[]
    for vent,label in [('RV',1),('LV',3)]:
        row=dict(ventricle=vent,native_volume_ml=float((mask==label).sum()*np.prod(spacing[:3])/1000))
        support=pro[:,::2,::2]
        row['softmax_entropy']=agg(entropy,struct_weight(support,label,False));row['measurement_aware_softmax_entropy']=agg(entropy,struct_weight(support,label,True))
        if a.reference:
            ref=np.asarray(nib.load(a.reference).dataobj,dtype=np.int16)
            if a.reference_labels=='mnms':ref=np.take(np.array([0,3,2,1],np.int16),ref)
            row['dice_processed']=dice_binary(pro==label,processed(ref,spacing[:2],True)==label)
        rows.append(row)
    import pandas as pd
    a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False,float_format='%.17g')

if __name__=='__main__':main()
