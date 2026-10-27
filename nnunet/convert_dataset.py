"""Original ACDC ED/ES images and original labels to nnU-Net v2 dataset 701."""
import argparse,json,shutil
from pathlib import Path

def convert(development_root,output):
    if any('refined' in part.lower() for part in Path(development_root).parts):raise ValueError('The nnU-Net baseline requires the original ACDC labels')
    paths=sorted(p for p in Path(development_root).glob('patient*') if (p/'Info.cfg').is_file())
    if len(paths)!=100:raise ValueError('Exactly the original 100-patient development partition is required')
    images=output/'imagesTr';labels=output/'labelsTr';images.mkdir(parents=True,exist_ok=True);labels.mkdir(exist_ok=True)
    for patient in paths:
        info={a.strip():b.strip() for a,b in (line.split(':',1) for line in (patient/'Info.cfg').read_text().splitlines() if ':' in line)}
        for phase in ['ED','ES']:
            stem=f'{patient.name}_frame{int(info[phase]):02d}'
            case=f'ACDC_{patient.name[-3:]}_{phase}'
            for src,dst in [(patient/(stem+'.nii.gz'),images/(case+'_0000.nii.gz')),(patient/(stem+'_gt.nii.gz'),labels/(case+'.nii.gz'))]:
                if dst.exists():raise FileExistsError('Refusing to overwrite converted data')
                shutil.copyfile(src,dst)
    dataset=json.loads((Path(__file__).parent/'dataset.json').read_text())
    (output/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();convert(a.development_root,a.output)

if __name__=='__main__':main()
