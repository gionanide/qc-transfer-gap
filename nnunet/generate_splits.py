"""Generate the frozen diagnosis-stratified, patient-grouped nnU-Net folds locally."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

def build_splits(development_root):
    paths=sorted(p for p in Path(development_root).glob('patient*') if (p/'Info.cfg').is_file())
    if len(paths)!=100:raise ValueError('The official development partition must contain exactly 100 patients')
    groups=[]
    for p in paths:
        info=dict(line.split(':',1) for line in (p/'Info.cfg').read_text().splitlines() if ':' in line)
        groups.append(info['Group'].strip())
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260724)
    splits=[]
    for train,val in splitter.split(np.arange(len(paths)),groups):
        assert not (set(train)&set(val))
        splits.append({role:[f'ACDC_{paths[int(i)].name[-3:]}_{phase}' for i in indices for phase in ['ED','ES']] for role,indices in [('train',train),('val',val)]})
    return splits

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(build_splits(a.development_root),indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
