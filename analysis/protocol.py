"""Fixed numerical settings and original seed recipes."""
import zlib
B=5000
BASE_SEED=20261023
ARCHS=['unet','segresnet','attnunet','swinunetr']
SCORES=['softmax_entropy','measurement_aware_softmax_entropy','mc_dropout','measurement_aware_mc_dropout']
BUDGETS=[0.,.05,.10,.15,.20,.30,.50]
ENDPOINTS=['EF_gt5','EF_gt10','relative_EDV_gt10pct']
ROBUST=['mean','median','trimmed_mean_10pct_each_tail']
GOALS=[('EF_gt5',10),('EF_gt5',20),('EF_gt10',10),('EF_gt10',20)]
def stable_seed(*parts):return int((20260827+zlib.crc32('|'.join(map(str,parts)).encode()))%(2**31-1))

def corr_seed(vent,cohort,arch,score,target):
    native={'dice':'one_minus_dice','EDV':'abs_delta_edv_ml' if vent=='RV' else 'abs_delta_lvedv_ml','SV':'abs_delta_sv_ml' if vent=='RV' else 'abs_delta_lvsv_ml','EF':'abs_delta_rvef_points' if vent=='RV' else 'abs_delta_lvef_points'}[target]
    if vent=='RV':return int((20260816+zlib.crc32('|'.join(['all',arch,score,native]).encode()))%(2**32-1))
    return stable_seed('lv_correlation','all',cohort,arch,score,native)

def gap_seed(vent,cohort,arch,score):return stable_seed('paired_gap',vent,cohort,arch,score)
def seed_for(sensitivity,cohort,ventricle,target):
    key='|'.join(['R1C1_RF_v1',sensitivity,cohort,ventricle,target])
    return int((BASE_SEED+zlib.crc32(key.encode('utf8')))%(2**32-1)),key
