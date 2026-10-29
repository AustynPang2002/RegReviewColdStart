"""Read archived applications, audit endpoint semantics and freeze temporal cohorts.

No approval ID, current outcome, current indication, price or reimbursement feature
is used as a predictor. Snapshots cannot establish historical field availability;
the resulting experiment is explicitly retrospective, not prospective.
"""
import argparse, io, json, re, unicodedata, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def normalize(x):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC',str(x or ''))).lower()

def labels_at(df, cutoff, cfg):
    cutoff=pd.Timestamp(cutoff)
    valid=df.start.le(cutoff)
    observed=df.event.gt(0)&df.end.notna()&df.end.le(cutoff)
    end=df.end.where(observed, cutoff)
    duration=(end-df.start).dt.days.clip(lower=0).to_numpy()
    ev=np.where(observed, df.event, 0).astype('int64')
    # Bins are [0,w),[w,2w),... . Censoring contributes only FULL observed bins.
    b=np.minimum(duration//cfg['bin_days'],cfg['bins']).astype('int64')
    ev=np.where(b>=cfg['bins'],0,ev)
    return duration.astype('int64'),ev,b,valid.to_numpy()

def past_features(df, half_life):
    """Only earlier accepted applications; no outcome labels in history.
    Same-day records do not see one another. Unknown names never form a group.
    """
    out=np.zeros((len(df),6),dtype='float32')
    states=[{},{}]
    decay=np.log(2)/half_life
    for day, group in df.groupby('start',sort=True):
        for idx,row in group.iterrows():
            for g,key in enumerate([row.drug_group,row.company_group]):
                if key and key in states[g]:
                    n,weighted,last=states[g][key]; gap=(day-last).days
                    out[idx,g*3:g*3+3]=[np.log1p(n),np.log1p(weighted*np.exp(-decay*gap)),np.log1p(gap)]
        for idx,row in group.iterrows():
            for g,key in enumerate([row.drug_group,row.company_group]):
                if not key: continue
                n,w,last=states[g].get(key,(0,0,day))
                states[g][key]=(n+1,w*np.exp(-decay*(day-last).days)+1,day)
    return out

def prepare(cfg):
    output=ROOT/'data'; output.mkdir(exist_ok=True)
    # Cache is generated locally from this exact archive; hash manifest documents it.
    with zipfile.ZipFile(cfg['archive']) as z:
        raw=z.read(cfg['member'])
    import hashlib
    digest=hashlib.sha256(raw).hexdigest()
    df=pd.read_excel(io.BytesIO(raw),dtype=str).fillna('')
    all_names=df.copy()
    total=len(df); audit={'raw_rows':total,'member_sha256':digest}
    audit['outcomes']=df['审评结论'].value_counts().to_dict()
    df['source_row']=np.arange(len(df))+2
    df['start']=pd.to_datetime(df['承办日期'],errors='coerce')
    df['end']=pd.to_datetime(df['审结日期'],errors='coerce')
    # Historical volumes must not select past records by their FUTURE outcomes.
    history_df=df[df.start.notna() & ~df['受理号'].duplicated() & df['受理号'].ne('')].copy().reset_index(drop=True)
    history_df['drug_group']=history_df['药品名称'].map(normalize)
    history_df['company_group']=history_df['企业名称'].map(normalize)
    history_values=past_features(history_df,cfg['history_half_life_days'])
    history_lookup={r:v for r,v in zip(history_df.source_row,history_values)}
    # Freeze only production/import requests; clinical permission is a different task.
    if cfg['cohort']=='production_import':
        df=df[df['申请事项'].isin(['S：申请生产','S：申请进口'])].copy()
    elif cfg['cohort']!='all': raise ValueError('Unknown cohort')
    audit['selected_cohort_rows']=len(df)
    duplicate=df['受理号'].duplicated(keep=False)|df['受理号'].eq('')
    audit['duplicate_or_empty_id_rows_excluded']=int(duplicate.sum())
    df=df[~duplicate].copy()
    df['event']=df['审评结论'].map(cfg['event_map']).fillna(0).astype(int)
    pending=df['审评结论'].isin(cfg['pending_values']) & df.end.isna()
    known=df.event.gt(0)&df.end.notna()
    plausible=df.start.notna() & (df.end.isna()|df.end.ge(df.start))
    eligible=plausible&(pending|known)
    excluded=df.loc[~eligible,['受理号','审评结论','承办日期','审结日期']]
    excluded.to_csv(output/'excluded_rows.csv',index=False,encoding='utf-8-sig')
    audit['unmapped_or_inconsistent_excluded']=len(excluded)
    # Unknown resolved outcomes are NOT assumed to be ongoing review.
    df=df[eligible].sort_values(['start','受理号']).reset_index(drop=True)
    inferred=max(df.start.max(),df.end.max())
    observation=pd.Timestamp(cfg['observation_end']) if cfg['observation_end'] else inferred
    if observation<pd.Timestamp(cfg['test_end']): raise ValueError('Observation end precedes test end')
    df=df[df.start.le(observation)].reset_index(drop=True)
    df['drug_group']=df['药品名称'].map(normalize)
    df['company_group']=df['企业名称'].map(normalize)
    df['split']=np.select([df.start.le(cfg['train_end']),df.start.le(cfg['val_end']),df.start.le(cfg['test_end'])],['train','val','test'],default='post_test')
    hist=np.stack([history_lookup[r] for r in df.source_row])
    numeric=[f'{g}_{f}' for g in ['drug','company'] for f in ['count_log','decayed_count_log','gap_log']]
    for i,c in enumerate(numeric): df[c]=hist[:,i]
    df['entry_year']=(df.start.dt.year-2000).astype(float); numeric+=['entry_year']
    # Novel relative to all pre-boundary applications, not just a training subsample.
    all_start=pd.to_datetime(all_names['承办日期'],errors='coerce')
    train_names=set(all_names.loc[all_start.le(cfg['train_end']),'药品名称'].map(normalize))
    train_companies=set(all_names.loc[all_start.le(cfg['train_end']),'企业名称'].map(normalize))
    val_names=set(all_names.loc[all_start.le(cfg['val_end']),'药品名称'].map(normalize))
    df['cold_drug']=~df.drug_group.isin(train_names)&df.drug_group.ne('')
    df['strict_cold_drug']=~df.drug_group.isin(val_names)&df.drug_group.ne('')
    df['cold_company']=~df.company_group.isin(train_companies)&df.company_group.ne('')
    for split,cutoff in [('train',cfg['train_end']),('val',cfg['val_end']),('test',observation),('post_test',observation)]:
        mask=df.split.eq(split); dur,event,b,_=labels_at(df[mask],cutoff,cfg)
        df.loc[mask,'duration']=dur; df.loc[mask,'target_event']=event; df.loc[mask,'time_bin']=b
    audit.update(eligible_rows=len(df),observation_end=str(observation.date()),observation_end_is_assumed=cfg['observation_end'] is None,
                 split_counts=df.split.value_counts().to_dict(),feature_numeric=numeric,
                 endpoint_counts=df.groupby('split').target_event.value_counts().unstack(fill_value=0).to_dict(),
                 cold_test=int((df.split.eq('test')&df.cold_drug).sum()),strict_cold_test=int((df.split.eq('test')&df.strict_cold_drug).sum()),
                 limitations=['name-group novelty is not validated active-ingredient novelty','current snapshot cannot prove historical field availability','pending status is provisional without last-contact timestamps'])
    df.to_pickle(output/'cohort.pkl')
    (output/'manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['outcomes','endpoint_counts']},ensure_ascii=False,indent=2))
    return df

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default=str(ROOT/'config.json')); a=p.parse_args()
    prepare(json.loads(Path(a.config).read_text(encoding='utf-8')))
