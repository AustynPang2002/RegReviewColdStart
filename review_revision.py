"""Reproducible reviewer-requested analyses, separate from archived experiments.

Run with the project virtual environment. No test-label model selection.
Bootstrap conditions on trained models and the training censoring estimator.
"""
import argparse, hashlib, json, pickle, time, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from metrics import CensorKM, horizon_targets, score
from prepare import labels_at

ROOT=Path(__file__).resolve().parent
HORIZONS=[180,360,720]
METRICS=['AUROC','AP','Brier']

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def text_matrix(df,cols):
    rows=[]; columns=[]; values=[]
    for i,parts in enumerate(df[cols].itertuples(index=False,name=None)):
        s=' '.join(map(str,parts))[:64]
        counts=Counter(1+int.from_bytes(hashlib.blake2b(c.encode(),digest_size=4).digest(),'little')%8191 for c in s)
        for c,n in counts.items(): rows.append(i); columns.append(c); values.append(n/max(1,len(s)))
    return sparse.csr_matrix((values,(rows,columns)),shape=(len(df),8192))

def fit_baselines(df,cfg,manifest,out):
    splits={s:df[df.split.eq(s)].copy() for s in ['train','val','test']}
    tr=splits['train']; km=CensorKM().fit(tr.duration,tr.target_event)
    cats=cfg['categorical_features']; nums=manifest['feature_numeric']
    one=OneHotEncoder(handle_unknown='ignore').fit(tr[cats])
    median=tr[nums].median().fillna(0); scale=StandardScaler().fit(tr[nums].fillna(median))
    # Train-only category vocabularies, explicit native categorical splits for HGB.
    ordinal=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan).fit(tr[cats])
    matrices={}; trees={}
    for s,d in splits.items():
        structured=sparse.hstack([one.transform(d[cats]),sparse.csr_matrix(scale.transform(d[nums].fillna(median)))],format='csr')
        matrices[s]=sparse.hstack([structured,text_matrix(d,cfg['text_features'])],format='csr')
        trees[s]=np.column_stack([ordinal.transform(d[cats]),d[nums].fillna(median).to_numpy()])
    records=[]; predictions=[]
    for h in HORIZONS:
        y,w,_=horizon_targets(tr.duration,tr.target_event,h,km); keep=w>0; yy=(y==1).astype(int)
        vy,vw,_=horizon_targets(splits['val'].duration,splits['val'].target_event,h,km)
        for name in ['matched_logistic','categorical_hgb']:
            best=None; best_loss=np.inf; candidates=[]; started=time.perf_counter()
            grid=[.01,.1,1.,10.] if name=='matched_logistic' else [(15,0.),(15,1.),(31,0.),(31,1.)]
            xx=matrices if name=='matched_logistic' else trees
            for param in grid:
                estimator=(LogisticRegression(C=param,solver='liblinear',max_iter=2000,random_state=42) if name=='matched_logistic' else HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=param[0],l2_regularization=param[1],categorical_features=list(range(len(cats))),early_stopping=False,random_state=42))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    estimator.fit(xx['train'][keep],yy[keep],sample_weight=w[keep])
                pp=estimator.predict_proba(xx['val'])[:,1]
                loss=float(np.sum(vw*((vy==1)-pp)**2)/len(vy))
                converged=not any(issubclass(x.category,ConvergenceWarning) for x in caught)
                candidates.append({'parameter':param,'validation_ipcw_brier':loss,'converged':converged})
                if not converged: raise RuntimeError('Unconverged baseline candidate')
                if loss<best_loss: best_loss=loss; best=estimator; selected=param
            elapsed=time.perf_counter()-started
            for s in ['val','test']:
                d=splits[s]; p=best.predict_proba(xx[s])[:,1]
                predictions.append(pd.DataFrame({'id':d['受理号'].astype(str),'split':s,'model':name,'horizon':h,'p':p}))
            records.append({'model':name,'horizon':h,'selected':selected,'candidates':candidates,'search_seconds':elapsed,'selection':'minimum validation IPCW Brier; first candidate wins ties','input_matched_to_full':name=='matched_logistic'})
            with (out/f'{name}_{h}.pkl').open('wb') as f:
                pickle.dump({'model':best,'onehot':one,'ordinal':ordinal,'scaler':scale,'median':median,'cats':cats,'nums':nums,'text_cols':cfg['text_features']},f)
            print('baseline',name,h,selected,round(elapsed,2),flush=True)
    save_json(out/'baseline_selection.json',records)
    pd.concat(predictions,ignore_index=True).to_csv(out/'baseline_predictions.csv',index=False)

def prediction_arrays(df,out):
    test=df[df.split.eq('test')].reset_index(drop=True); ids=test['受理号'].astype(str)
    arrays={}
    for name in ['ft_base','full','no_text','no_history','no_decay','no_multitask']:
        runs=[]
        for seed in range(42,47):
            p=pd.read_csv(ROOT/f'results/server01_ablation_5seeds/{name}_seed{seed}/predictions.csv',dtype={'受理号':str})
            p=p[p.split.eq('test')].set_index('受理号')
            if not p.index.is_unique or set(p.index)!=set(ids): raise ValueError('Prediction/cohort ID mismatch')
            p=p.loc[ids]
            if not np.allclose(p.duration,test.duration) or not np.array_equal(p.target_event.to_numpy(),test.target_event.to_numpy()): raise ValueError('Archived label mismatch')
            runs.append(p[[f'p_approval_{h}d' for h in HORIZONS]].to_numpy())
        arrays[name]=np.stack(runs)
    new=pd.read_csv(out/'baseline_predictions.csv',dtype={'id':str}); new=new[new.split.eq('test')]
    for name in new.model.unique():
        arrays[name]=np.stack([new[(new.model==name)&(new.horizon==h)].set_index('id').loc[ids,'p'].to_numpy() for h in HORIZONS],axis=-1)[None]
    # Recompute old references from saved fitted pipelines; do not refit them.
    for name in ['logistic','hist_gradient_boosting']:
        ps=[]
        for h in HORIZONS:
            with (ROOT/f'runs/baselines/{name}_{h}.pkl').open('rb') as f: pre,est=pickle.load(f)
            ps.append(est.predict_proba(pre.transform(test))[:,1])
        arrays['archived_'+name]=np.stack(ps,-1)[None]
    return test,arrays

class FastMetric:
    """Exact weighted ROC/AP including ties; pre-sort once for bootstrap."""
    def __init__(self,y,w,p):
        self.y=np.asarray(y)==1; self.w=np.asarray(w); self.p=np.asarray(p)
        self.order=np.argsort(-self.p,kind='stable')
        sp=self.p[self.order]; self.ends=np.r_[np.flatnonzero(np.diff(sp)!=0),len(sp)-1]
    def __call__(self,multiplicity):
        w=self.w*multiplicity; order=self.order
        tp=np.cumsum(w[order]*self.y[order])[self.ends]
        fp=np.cumsum(w[order]*(~self.y[order]))[self.ends]
        brier=np.sum(w*(self.y-self.p)**2)/np.sum(multiplicity)
        if tp[-1]<=0 or fp[-1]<=0: return np.array([np.nan,np.nan,brier])
        t=np.r_[0,tp]/tp[-1]; f=np.r_[0,fp]/fp[-1]
        auc=np.sum(np.diff(f)*(t[1:]+t[:-1])*.5)
        ap=np.sum(np.diff(np.r_[0,tp])/tp[-1]*np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0))
        return np.array([auc,ap,brier])

def bootstrap_analysis(test,arrays,km,out,n_boot):
    rows=[]; diffs=[]
    groups={'all':np.ones(len(test),bool),'strict_cold_drug':test.strict_cold_drug.to_numpy(),'new_drug_application':test['申请类型（CDE）'].eq('新药').to_numpy()}
    pairs=[('full','ft_base'),('full','matched_logistic'),('full','categorical_hgb'),('no_history','full'),('ft_base','matched_logistic'),('full','archived_logistic'),('full','archived_hist_gradient_boosting')]
    for group,mask in groups.items():
        d=test[mask]; labels=d.drug_group.to_numpy(); unique,inverse=np.unique(labels,return_inverse=True)
        # A blank name, if present, is not treated as a shared biological cluster.
        if '' in unique: raise ValueError('Missing drug names require explicit singleton clusters')
        for j,h in enumerate(HORIZONS):
            y,w,_=horizon_targets(d.duration,d.target_event,h,km)
            metrics={name:[FastMetric(y,w,p[mask,j]) for p in ps] for name,ps in arrays.items()}
            points={name:np.mean([fn(np.ones(len(d))) for fn in fns],0) for name,fns in metrics.items()}
            draws={name:np.empty((n_boot,3)) for name in metrics}; rng=np.random.default_rng(20260925)
            for b in range(n_boot):
                counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))[inverse]
                for name,fns in metrics.items(): draws[name][b]=np.mean([fn(counts) for fn in fns],0)
            for name in metrics:
                for k,metric in enumerate(METRICS):
                    good=np.isfinite(draws[name][:,k]); ci=np.quantile(draws[name][good,k],[.025,.975]) if good.any() else [np.nan,np.nan]
                    rows.append(dict(group=group,horizon=h,model=name,metric=metric,estimate=points[name][k],lower=ci[0],upper=ci[1],valid_bootstrap=int(good.sum()),requested_bootstrap=n_boot,n=len(d),clusters=len(unique),positives=int(((y==1)&(w>0)).sum()),seeds=len(metrics[name])))
            for a,b in pairs:
                for k,metric in enumerate(METRICS):
                    delta=draws[a][:,k]-draws[b][:,k]; good=np.isfinite(delta); ci=np.quantile(delta[good],[.025,.975]) if good.any() else [np.nan,np.nan]
                    diffs.append(dict(group=group,horizon=h,comparison=f'{a} - {b}',metric=metric,difference=points[a][k]-points[b][k],lower=ci[0],upper=ci[1],valid_bootstrap=int(good.sum()),requested_bootstrap=n_boot))
            print('bootstrap',group,h,flush=True)
    pd.DataFrame(rows).to_csv(out/'cluster_bootstrap_metrics.csv',index=False)
    pd.DataFrame(diffs).to_csv(out/'paired_differences.csv',index=False)

def audits(df,cfg,manifest,test,arrays,km,out):
    collision={}
    for split in ['train','val','test','all_analysis']:
        d=df[df.split.isin(['train','val','test'])] if split=='all_analysis' else df[df.split.eq(split)]
        vocab=set(''.join(' '.join(map(str,row))[:64] for row in d[cfg['text_features']].itertuples(index=False,name=None)))
        buckets=defaultdict(list)
        for ch in sorted(vocab): buckets[1+int.from_bytes(hashlib.blake2b(ch.encode(),digest_size=4).digest(),'little')%8191].append(ch)
        collided=[v for v in buckets.values() if len(v)>1]; n=sum(map(len,collided))
        collision[split]={'unique_characters':len(vocab),'occupied_buckets':len(buckets),'collision_buckets':len(collided),'characters_in_collision_buckets':n,'fraction_characters_in_collision_buckets':n/len(vocab),'excess_characters':len(vocab)-len(buckets),'collisions':collided}
    save_json(out/'hash_collisions.json',collision)
    nums=manifest['feature_numeric']; a=[]
    for group,d in [('train',df[df.split.eq('train')]),('test_all',test),('strict_cold',test[test.strict_cold_drug]),('test_other',test[~test.strict_cold_drug])]:
        for c in nums:
            a.append(dict(group=group,feature=c,n=len(d),missing=int(d[c].isna().sum()),zero_fraction=float(d[c].eq(0).mean()),mean=float(d[c].mean()),median=float(d[c].median()),q25=float(d[c].quantile(.25)),q75=float(d[c].quantile(.75))))
    pd.DataFrame(a).to_csv(out/'feature_history_audit.csv',index=False)
    sens=[]
    # Scenarios are transparent calendar landmarks, NOT asserted extraction dates.
    for cutoff in ['2024-12-31','2025-06-30','2025-12-31','2026-05-19']:
        duration,event,_,valid=labels_at(test,cutoff,cfg)
        for j,h in enumerate(HORIZONS):
            for population,mask in [('all',valid),('mature_at_horizon',valid&((pd.Timestamp(cutoff)-test.start).dt.days.to_numpy()>=h))]:
                if not mask.any(): continue
                for name in ['full','ft_base','matched_logistic','categorical_hgb']:
                    scores=[]
                    for p in arrays[name]:
                        pp=p[mask,j]; scores.append(score(duration[mask],event[mask],np.stack([1-pp,pp,pp*0,pp*0],-1),h,km))
                    row=dict(cutoff=cutoff,population=population,horizon=h,model=name,n=int(mask.sum()),evaluated=scores[0]['evaluated'],positives=scores[0]['observed_approvals_before_horizon'],support_excluded=scores[0]['censoring_support_excluded'])
                    for field in ['ipcw_auroc_approval','ipcw_auprc_approval','ipcw_brier_approval','ipcw_mass']: row[field]=float(np.mean([s.get(field,np.nan) for s in scores]))
                    sens.append(row)
    pd.DataFrame(sens).to_csv(out/'cutoff_sensitivity.csv',index=False)
    save_json(out/'audit_metadata.json',{'observation_end_verified':False,'field_point_in_time_verified':False,'cutoff_interpretation':'hypothetical administrative recensoring, fixed models and training KM; cannot reconstruct historical database membership or last contact','numeric_missing_all_analysis':df[df.split.isin(['train','val','test'])][nums].isna().sum().to_dict(),'cohort_sha256':hashlib.sha256((ROOT/'data/cohort.pkl').read_bytes()).hexdigest()})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--reuse-baselines',action='store_true'); p.add_argument('--output',default='revision_20260925'); args=p.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_pickle(ROOT/'data/cohort.pkl'); cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8')); manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    tr=df[df.split.eq('train')]; km=CensorKM().fit(tr.duration,tr.target_event)
    with threadpool_limits(limits=4):
        if not args.reuse_baselines: fit_baselines(df,cfg,manifest,out)
        test,arrays=prediction_arrays(df,out)
        audits(df,cfg,manifest,test,arrays,km,out)
        bootstrap_analysis(test,arrays,km,out,args.bootstrap)
    save_json(out/'analysis_protocol.json',{'bootstrap_replicates':args.bootstrap,'seed':20260925,'resampling_unit':'normalized drug-name cluster within each reported population','estimand':'mean of per-seed metrics, NOT metric of ensemble predictions','paired':True,'training_censoring_KM':'fixed','training_samples_and_model_parameters':'fixed; training-sample uncertainty not included','multiplicity_correction':False,'status':'exploratory post-review analysis','horizons_days':HORIZONS,'baseline_selection':'validation IPCW Brier, separately per horizon; four candidates per family','neural_retuning':False})

if __name__=='__main__': main()
