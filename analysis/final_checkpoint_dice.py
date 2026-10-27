"""Supplementary Table S11: final-checkpoint Dice on the seed-42 80/20 development split."""
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import numpy as np
import nibabel as nib
import torch
from dataset import stratified_split,parse_info_cfg,frame_file,_resample_inplane,_crop_or_pad,_normalize
from model_zoo import build_model
from .io import low_priority,save_rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-root',required=True,type=Path);p.add_argument('--weights',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--include-test-root',type=Path);a=p.parse_args();low_priority()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    train,val=stratified_split(a.development_root,.20,42)
    parts={'train':train,'validation':val}
    if a.include_test_root:parts['test']=sorted(str(p) for p in a.include_test_root.glob('patient*') if (p/'Info.cfg').is_file())
    names={'unet':'last.pt','segresnet':'last_segresnet.pt','attnunet':'last_attnunet.pt','swinunetr':'last_swinunetr.pt'}
    rows=[]
    for arch,name in names.items():
        cp=a.weights/name;digest=hashlib.sha256(cp.read_bytes()).hexdigest();net=build_model(arch);net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);net.eval()
        for part,patients in parts.items():
            values={label:[] for label in [1,2,3]}
            for directory in patients:
                directory=Path(directory);info=parse_info_cfg(directory);phase_values={label:[] for label in values}
                for phase in ['ED','ES']:
                    image=nib.load(frame_file(str(directory),directory.name,info[phase]));ref=np.asarray(nib.load(frame_file(str(directory),directory.name,info[phase],True)).dataobj)
                    raw=np.asarray(image.dataobj,dtype=np.float32);spacing=image.header.get_zooms()[:2];pred=[];gt=[]
                    with torch.no_grad():
                        for z in range(raw.shape[2]):
                            x=_normalize(_crop_or_pad(_resample_inplane(raw[:,:,z],spacing,(1.25,1.25),False),(256,256),0))
                            pred.append(net(torch.from_numpy(x)[None,None]).argmax(1).numpy()[0]);gt.append(_crop_or_pad(_resample_inplane(ref[:,:,z],spacing,(1.25,1.25),True),(256,256),0))
                    pred=np.stack(pred);gt=np.stack(gt)
                    for label in values:
                        aa=pred==label;bb=gt==label;den=aa.sum()+bb.sum();phase_values[label].append(2*np.logical_and(aa,bb).sum()/den if den else np.nan)
                for label in values:values[label].append(float(np.mean(phase_values[label])))
            for label,structure in [(1,'RV'),(2,'MYO'),(3,'LV')]:rows.append(dict(architecture=arch,partition=part,n_patients=len(patients),structure=structure,mean_patient_Dice=float(np.mean(values[label])),checkpoint_sha256=digest,split='reconstructed_seed42_80_20; historical_identity_unverified'))
    save_rows(a.output,rows)

if __name__=='__main__':main()
