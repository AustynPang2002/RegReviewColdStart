"""Predict on the frozen, audited cohort using its same as-of feature builder.
For new raw submissions, rerun prepare on an updated archive before inference.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import RegReviewModel,probabilities
from prepare import past_features,normalize
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/main');p.add_argument('--split',default='post_test',choices=['train','val','test','post_test']);p.add_argument('--input-csv');p.add_argument('--output',default='runs/inference.csv');a=p.parse_args()
    run=ROOT/a.run; checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    with (run/'features.pkl').open('rb') as f: features=pickle.load(f)
    m=RegReviewModel(checkpoint['n_numeric'],checkpoint['cardinalities'],checkpoint['bins'],no_text=checkpoint['no_text']);m.load_state_dict(checkpoint['state_dict']);m.eval();torch.set_num_threads(4)
    if a.input_csv:
        import io,zipfile
        query=pd.read_csv(a.input_csv,dtype=str).fillna('')
        required=['受理号','药品名称','承办日期','企业名称']
        if not set(required).issubset(query):raise ValueError('Required columns: '+','.join(required))
        if query['受理号'].duplicated().any():raise ValueError('Duplicate query application IDs')
        for col in cfg['categorical_features']:
            if col not in query:query[col]=''
        if not query['申请事项'].isin(['S：申请生产','S：申请进口']).all():raise ValueError('Only production/import applications supported')
        with zipfile.ZipFile(cfg['archive']) as z:
            hist=pd.read_excel(io.BytesIO(z.read(cfg['member'])),dtype=str).fillna('')
        hist=hist[~hist['受理号'].isin(query['受理号']) & hist['受理号'].ne('')].drop_duplicates('受理号').copy()
        hist['_query']=False;query['_query']=True
        combined=pd.concat([hist,query],ignore_index=True)
        combined['start']=pd.to_datetime(combined['承办日期'],errors='coerce')
        if combined.loc[combined._query,'start'].isna().any():raise ValueError('Unparseable query date')
        combined=combined[combined.start.notna()].reset_index(drop=True)
        combined['drug_group']=combined['药品名称'].map(normalize);combined['company_group']=combined['企业名称'].map(normalize)
        arr=past_features(combined,cfg['history_half_life_days'])
        for i,c in enumerate([f'{g}_{f}' for g in ['drug','company'] for f in ['count_log','decayed_count_log','gap_log']]):combined[c]=arr[:,i]
        combined['entry_year']=(combined.start.dt.year-2000).astype(float)
        df=combined[combined._query].copy()
    else:
        df=pd.read_pickle(ROOT/'data/cohort.pkl');df=df[df.split.eq(a.split)].copy()
    preds=[]
    with torch.no_grad():
        for start in range(0,len(df),256):
            logits,_=m(*[torch.from_numpy(x) for x in features.transform(df.iloc[start:start+256])]);s,c=probabilities(logits)
            indices=[h//cfg['bin_days']-1 for h in [180,360,720]]
            preds.append(torch.cat([s[:,indices,None],c[:,indices]],-1).numpy())
    if not preds: raise ValueError('No rows in requested split')
    arr=np.concatenate(preds);out=df[['受理号','药品名称','承办日期']].copy()
    for j,h in enumerate([180,360,720]):
        for k,label in enumerate(['pending','approval','rejection','withdrawal']):out[f'p_{label}_{h}d']=arr[:,j,k]
    path=ROOT/a.output;path.parent.mkdir(parents=True,exist_ok=True);out.to_csv(path,index=False,encoding='utf-8-sig');print(path)
if __name__=='__main__': main()
