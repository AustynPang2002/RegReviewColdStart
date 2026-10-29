"""IPCW weighted horizon logistic regression and histogram boosting.
Uses identical cohorts, horizon definitions and training-only encoders.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from metrics import CensorKM,horizon_targets,score
ROOT=Path(__file__).resolve().parent
def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--max-per-split',type=int,default=0); pa.add_argument('--seed',type=int,default=42); args=pa.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8')); manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    df=pd.read_pickle(ROOT/'data/cohort.pkl'); splits={s:df[df.split.eq(s)].copy() for s in ['train','val','test']}
    for s,d in splits.items():
        if args.max_per_split and len(d)>args.max_per_split: splits[s]=d.sample(args.max_per_split,random_state=args.seed)
    cats=cfg['categorical_features']; nums=manifest['feature_numeric']; train=splits['train']
    km=CensorKM().fit(train.duration,train.target_event); report={}; output=ROOT/'runs/baselines'; output.mkdir(parents=True,exist_ok=True)
    for h in [180,360,720]:
        y,w,_=horizon_targets(train.duration,train.target_event,h,km); keep=w>0; y=(y==1).astype(int)
        for name in ['logistic','hist_gradient_boosting']:
            if name=='logistic':
                pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',StandardScaler(),nums)])
                estimator=LogisticRegression(max_iter=500,C=1)
            else:
                pre=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),cats),('num','passthrough',nums)])
                estimator=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=args.seed,early_stopping=False)
            # Fixed baseline parameters; no fitting or tuning on test labels.
            xx=pre.fit_transform(train.loc[keep]); estimator.fit(xx,y[keep],sample_weight=w[keep])
            for s in ['val','test']:
                d=splits[s]; p=estimator.predict_proba(pre.transform(d))[:,1]; probs=np.stack([1-p,p,np.zeros_like(p),np.zeros_like(p)],-1)
                for group,mask in {'all':np.ones(len(d),bool),'strict_cold_drug':d.strict_cold_drug.to_numpy()}.items():
                    if mask.any(): report[f'{name}/{s}/{group}/{h}']=score(d.duration.to_numpy()[mask],d.target_event.to_numpy()[mask],probs[mask],h,km)
            with (output/f'{name}_{h}.pkl').open('wb') as f: pickle.dump((pre,estimator),f)
            print(name,h,flush=True)
    (output/'metrics.json').write_text(json.dumps({'arguments':vars(args),'status':'smoke_test' if args.max_per_split else 'experiment','scores':report},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
