"""nnU-Net training entry point with the recorded fold seed; never invoked by packaging."""
import argparse,random
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--device',default='cpu',choices=['cpu','cuda']);a=p.parse_args()
    seed=20260804+a.fold;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    from nnunetv2.run.run_training import run_training
    run_training('701','2d',a.fold,trainer_class_name='QualificationNNUNetTrainer',plans_identifier='nnUNetPlans',device=torch.device(a.device))

if __name__=='__main__':main()
