"""Explicitly register the portable trainer in the selected nnU-Net environment."""
import argparse,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');a=p.parse_args()
    import nnunetv2
    destination=Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/QualificationNNUNetTrainer.py'
    if not a.install:
        print('Pass --install to register the trainer in the active nnU-Net environment.');return
    if destination.exists():raise FileExistsError('Trainer already exists; inspect it before replacing anything')
    shutil.copyfile(Path(__file__).parent/'QualificationNNUNetTrainer.py',destination)

if __name__=='__main__':main()
