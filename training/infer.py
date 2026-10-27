"""Fixed-checkpoint phase inference, native volumes, QC and twenty MC passes."""
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
import argparse,hashlib,json,time,zlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from model_zoo import build_model
from geometry import preprocess_volume,processed_prediction_to_native
from score_functions import struct_weight,agg,enable_dropout
from entropy import deterministic_entropy

def entropy_scores(entropy,mask):
    support=mask[:,::2,::2][:,:128,:128]
    return {(vent,aware):agg(entropy,struct_weight(support,label,aware)) for vent,label in [('RV',1),('LV',3)] for aware in [False,True]}

def infer(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device);model=build_model(a.architecture);model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True);model.to(device).eval()
    nii=nib.load(a.image);raw=np.asarray(nii.dataobj,dtype=np.float32)
    if raw.ndim==4:
        if a.frame is None:raise ValueError('A zero-based frame index is required for 4D input')
        raw=raw[...,a.frame]
    sx,sy,sz=nii.header.get_zooms()[:3];images,_,shapes=preprocess_volume(raw,np.zeros(raw.shape,np.int16),float(sx),float(sy))
    a.output.mkdir(parents=True,exist_ok=True)
    batch=len(images) if a.batch_slices==0 else a.batch_slices
    predicted=[];entropies=[]
    with torch.no_grad():
        for lo in range(0,len(images),batch):
            logits=model(torch.from_numpy(images[lo:lo+batch,None]).to(device));probs=torch.softmax(logits,1)
            predicted.append(logits.argmax(1).cpu().numpy().astype(np.int16))
            entropies.append(deterministic_entropy(probs.cpu().numpy()))
    predicted=np.concatenate(predicted);entropy=np.concatenate(entropies);native=processed_prediction_to_native(predicted,shapes,raw.shape)
    base=dict(cohort=a.cohort,architecture=a.architecture,patient_id=a.patient_id,phase=a.phase)
    support=predicted
    gate={}
    if a.stored_mask:
        support=np.load(a.stored_mask,allow_pickle=False)
        if support.shape!=predicted.shape:raise ValueError('Stored processed mask shape differs')
        changed=int(np.count_nonzero(support!=predicted));fraction=changed/predicted.size
        gate.update(changed_voxels=changed,changed_fraction=fraction)
        if fraction>.001:raise RuntimeError('Deterministic phase differs by more than 0.1% of voxels')
        if a.stored_entropy:
            old=np.load(a.stored_entropy,allow_pickle=False)
            old_scores=entropy_scores(old,support);new_scores=entropy_scores(entropy,support)
            new_support_scores=entropy_scores(entropy,predicted)
            gate['score_changes_from_mask_flips']={v:{'plain':new_support_scores[v,False]-new_scores[v,False],'measurement_aware':new_support_scores[v,True]-new_scores[v,True]} for v in ['RV','LV']}
            difference=max(abs(old_scores[v,False]-new_scores[v,False]) for v in ['RV','LV'])
            gate['stored_mask_entropy_difference']=difference
            if not np.isfinite(difference) or difference>1e-5:raise RuntimeError('Stored-mask entropy check failed')
    scores=entropy_scores(entropy,support);rows=[];passes=[]
    for vent,label in [('RV',1),('LV',3)]:rows.append(dict(base,ventricle=vent,native_volume_ml=float((native==label).sum()*sx*sy*sz/1000),softmax_entropy=scores[vent,False],measurement_aware_softmax_entropy=scores[vent,True]))
    if a.save_masks:
        nib.save(nib.Nifti1Image(native,nii.affine),a.output/'native_mask.nii.gz')
        np.save(a.output/'processed_mask.npy',predicted,allow_pickle=False)
    seed=int((20260816+zlib.crc32(f'{a.architecture}|{a.patient_id}|{a.phase}|mc_dropout_t20'.encode()))%(2**31-1))
    if a.mc:
        model.eval();enable_dropout(model);torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        # Twenty CPU probability maps on the exact saved stride-two score grid.
        probabilities=np.empty((20,len(images),4,128,128),np.float32);masks=np.empty((20,len(images),256,256),np.uint8)
        with torch.no_grad():
            if a.loop=='slice-major':
                for s in range(len(images)):
                    x=torch.from_numpy(images[s:s+1,None]).to(device)
                    for i in range(20):
                        logits=model(x);pr=torch.softmax(logits,1);masks[i,s]=logits.argmax(1).cpu().numpy()[0];probabilities[i,s]=pr[:,:,::2,::2].cpu().numpy()[0]
                    time.sleep(a.pause)
            else:
                for i in range(20):
                    for lo in range(0,len(images),batch):
                        logits=model(torch.from_numpy(images[lo:lo+batch,None]).to(device));pr=torch.softmax(logits,1)
                        masks[i,lo:lo+batch]=logits.argmax(1).cpu().numpy();probabilities[i,lo:lo+batch]=pr[:,:,::2,::2].cpu().numpy()
                    time.sleep(a.pause)
        mean=torch.from_numpy(probabilities).mean(0);mc_entropy=(-(mean*mean.clamp_min(1e-8).log()).sum(1)).numpy().astype(np.float64)
        mc_scores=entropy_scores(mc_entropy,support)
        for row in rows:
            row['new_mc_dropout']=mc_scores[row['ventricle'],False];row['new_measurement_aware_mc_dropout']=mc_scores[row['ventricle'],True]
        for i,mask in enumerate(masks):
            restored=processed_prediction_to_native(mask,shapes,raw.shape);digest=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
            for vent,label in [('RV',1),('LV',3)]:passes.append(dict(base,ventricle=vent,pass_ordinal=i+1,native_volume_ml=float((restored==label).sum()*sx*sy*sz/1000),processed_mask_sha256=digest))
        pd.DataFrame(passes).to_csv(a.output/'mc_native_volumes.csv',index=False,float_format='%.17g')
    pd.DataFrame(rows).to_csv(a.output/'phase_measurements_and_scores.csv',index=False,float_format='%.17g')
    (a.output/'inference_settings.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),MC_seed=seed,MC_passes=20 if a.mc else 0,loop=a.loop,batch_slices=batch,score_grid='stride-two',dropout=.1,deterministic_gate=gate),indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True);p.add_argument('--frame',type=int);p.add_argument('--checkpoint',required=True);p.add_argument('--architecture',required=True,choices=['unet','segresnet','attnunet','swinunetr']);p.add_argument('--patient-id',required=True);p.add_argument('--cohort',required=True);p.add_argument('--phase',required=True,choices=['ED','ES']);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--mc',action='store_true');p.add_argument('--loop',choices=['slice-major','phase-batched'],default='slice-major');p.add_argument('--batch-slices',type=int,default=1);p.add_argument('--pause',type=float,default=.075);p.add_argument('--save-masks',action='store_true');p.add_argument('--stored-mask');p.add_argument('--stored-entropy');infer(p.parse_args())

if __name__=='__main__':main()
