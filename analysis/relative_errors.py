"""Table S10: absolute/relative EDV and SV correlations, gaps and recovered fractions."""
from .io import parser,load,low_priority
from .reference import compute

def main():
    a=parser(__doc__).parse_args();low_priority();d=load(a.input,a.sensitivity)
    compute(d,a.output/'absolute',a.draws,a.scores,False)
    compute(d,a.output/'relative',a.draws,a.scores,True)
    from .io import save_rows
    save_rows(a.output/'relative_SV_exclusions.csv',[dict(cohort=c,ventricle=v,architecture=m,nonpositive_reference_SV=int((g.ref_sv_ml<=0).sum()),undefined_ratio=int((~__import__('numpy').isfinite(g.relative_SV)).sum())) for (c,v,m),g in d.groupby(['cohort','ventricle','architecture'])])

if __name__=='__main__':main()
