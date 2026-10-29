"""Read-only analyses of frozen models/cohorts for the second review. No refitting."""
from pathlib import Path
import sys, json, pickle, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--output',type=Path,default=Path('results/final_diagnostics'))
args=parser.parse_args()
ROOT=args.project_root.resolve()
sys.path.insert(0,str(ROOT))
from metrics import CensorKM, horizon_targets
from prepare import labels_at
from review_revision import text_matrix
OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
df=pd.read_pickle(ROOT/'data/cohort.pkl')
raw=pd.read_pickle(ROOT/'data/raw_applications.pkl').fillna('')
raw=raw[raw['申请事项'].isin(['S：申请生产','S：申请进口'])].copy()
raw['start']=pd.to_datetime(raw['承办日期'],errors='coerce')
raw['end']=pd.to_datetime(raw['审结日期'],errors='coerce')
raw['split']=np.select([raw.start.isna(),raw.start.le('2020-12-31'),raw.start.le('2022-12-31'),raw.start.le('2024-12-31')],['Unknown','Train','Validation','Test'],default='Post-test')
dup=raw['受理号'].duplicated(keep=False)|raw['受理号'].eq('')
mapped=raw['审评结论'].isin(cfg['event_map']); pending=raw['审评结论'].isin(cfg['pending_values'])
raw['reason']=np.select([dup,raw.start.isna(),raw.end.lt(raw.start),mapped&raw.end.isna(),pending&raw.end.notna(),~mapped&~pending],['Missing/duplicate ID','Invalid start','End before start','Mapped outcome without end','Pending/blank with end','Unmapped outcome'],default='Retained')
counts=pd.crosstab(raw.reason,raw.split).reindex(columns=['Train','Validation','Test','Post-test','Unknown'],fill_value=0)
counts['Total']=counts.sum(axis=1);counts.to_csv(OUT/'eligibility_by_reason.csv')
assert counts.loc['Retained','Total']==90580
assert counts.Total.sum()==98968
assert set(raw.loc[raw.reason.eq('Retained'),'受理号'])==set(df['受理号'])
print('ELIGIBILITY\n',counts.to_string(),flush=True)
tr=df[df.split.eq('train')]; km=CensorKM().fit(tr.duration,tr.target_event)
new=pd.read_csv(ROOT/'revision_20260925/baseline_predictions.csv',dtype={'id':str})
models=['matched_logistic','categorical_hgb','ft_base','full']
splits={s:df[df.split.eq(s)].reset_index(drop=True) for s in ['train','val','test']}
arrays={}
for s,d in splits.items():
 ids=d['受理号'].astype(str)
 arrays[s]={}
 for name in models:
  if name in ['ft_base','full']:
   if s=='train': continue
   runs=[]
   for seed in range(42,47):
    p=pd.read_csv(ROOT/f'results/server01_ablation_5seeds/{name}_seed{seed}/predictions.csv',dtype={'受理号':str})
    p=p[p.split.eq(s)].set_index('受理号').loc[ids]
    assert np.array_equal(p.target_event.to_numpy(),d.target_event.to_numpy())
    assert np.allclose(p.duration,d.duration)
    runs.append(p[[f'p_approval_{h}d' for h in [180,360,720]]].to_numpy())
   arrays[s][name]=np.stack(runs)
  elif s!='train':
   arrays[s][name]=np.stack([new[(new.model==name)&(new.split==s)&(new.horizon==h)].set_index('id').loc[ids,'p'].to_numpy() for h in [180,360,720]],-1)[None]
  else:
   ps=[]
   for h in [180,360,720]:
    with (ROOT/f'revision_20260925/{name}_{h}.pkl').open('rb') as f: b=pickle.load(f)
    if name=='matched_logistic':
     x=sparse.hstack([b['onehot'].transform(d[b['cats']]),sparse.csr_matrix(b['scaler'].transform(d[b['nums']].fillna(b['median']))),text_matrix(d,b['text_cols'])],format='csr')
    else: x=np.column_stack([b['ordinal'].transform(d[b['cats']]),d[b['nums']].fillna(b['median']).to_numpy()])
    ps.append(b['model'].predict_proba(x)[:,list(b['model'].classes_).index(1)])
    if name=='matched_logistic' and h==720:
     j=b['onehot'].transform(d[b['cats']]).shape[1]+b['nums'].index('entry_year')
     print('720 LR year log-odds per year',b['model'].coef_[0,j]/b['scaler'].scale_[b['nums'].index('entry_year')],flush=True)
     (OUT/'lr720_model_audit.json').write_text(json.dumps({'classes':b['model'].classes_.tolist(),'entry_year_coefficient_per_calendar_year':float(b['model'].coef_[0,j]/b['scaler'].scale_[b['nums'].index('entry_year')]),'converged_iterations':b['model'].n_iter_.tolist(),'C':b['model'].C},indent=2))
   arrays[s][name]=np.stack(ps,-1)[None]

support=[]; diagnostics=[]; bins=[]; mature=[]
def summarize(y,w,p):
 keep=w>0; yy=(y[keep]==1).astype(float); pp=p[keep]; ww=w[keep]
 return {'brier':float(np.sum(ww*(yy-pp)**2)/len(y)), 'prevalence':float(np.average(yy,weights=ww)), 'predicted':float(np.average(pp,weights=ww)), 'mass':float(w.mean()), 'auroc':float(roc_auc_score(yy,pp,sample_weight=ww)), 'ap':float(average_precision_score(yy,pp,sample_weight=ww))}
def calibrate(y,w,p):
 keep=w>0; yy=(y[keep]==1).astype(float); pp=p[keep]; ww=w[keep]; ww=ww/ww.sum(); lp=logit(np.clip(pp,1e-6,1-1e-6))
 def obj(a,offset):
  z=lp+a[0] if offset else a[0]+a[1]*lp
  return np.sum(ww*(np.logaddexp(0,z)-yy*z))
 intercept=minimize(obj,[0.],args=(True,),method='BFGS')
 slope=minimize(obj,[0.,1.],args=(False,),method='BFGS')
 return float(intercept.x[0]),float(slope.x[0]),float(slope.x[1])
for s,d in splits.items():
 own=CensorKM().fit(d.duration,d.target_event)
 for hidx,h in enumerate([180,360,720]):
  for estimator,k in [('training_KM',km),('cohort_KM',own)]:
   y,w,u=horizon_targets(d.duration,d.target_event,h,k); keep=w>0
   support.append(dict(split=s,horizon=h,estimator=estimator,n=len(d),evaluable=int(keep.sum()),approvals=int(((y==1)&keep).sum()),G_h=float(k.at(h,left=True)),mass=float(w.mean()),max_weight=float(w.max()),ess=float(w.sum()**2/np.sum(w*w)),support_excluded=int(u.sum()),prevalence=float(np.average(y[keep]==1,weights=w[keep]))))
   for name,ps in arrays[s].items():
    for seed,pall in enumerate(ps):
     p=pall[:,hidx]; v=summarize(y,w,p); ci,ca,cs=calibrate(y,w,p)
     row=dict(split=s,horizon=h,estimator=estimator,model=name,seed=seed+42,**v,calibration_intercept_offset=ci,calibration_joint_intercept=ca,calibration_slope=cs,p10=float(np.quantile(p,.1)),p50=float(np.median(p)),p90=float(np.quantile(p,.9)),positive_brier=float(np.sum(w*(y==1)*(1-p)**2)/len(y)),negative_brier=float(np.sum(w*(y!=1)*p*p)/len(y)))
     diagnostics.append(row)
     if s=='test' and estimator=='training_KM':
      # Equal-count bins among evaluable observations, with IPCW summaries.
      ix=np.flatnonzero(keep); ix=ix[np.argsort(p[ix],kind='stable')]
      for j,bix in enumerate(np.array_split(ix,10)):
       bins.append(dict(model=name,seed=seed+42,horizon=h,bin=j+1,n=len(bix),weight=float(w[bix].sum()),predicted=float(np.average(p[bix],weights=w[bix])),observed=float(np.average(y[bix]==1,weights=w[bix]))))
for cutoff in ['2024-12-31','2025-06-30','2025-12-31','2026-05-19']:
 d=splits['test']; duration,event,_,valid=labels_at(d,cutoff,cfg)
 for j,h in enumerate([180,360,720]):
  mask=valid&((pd.Timestamp(cutoff)-d.start).dt.days.to_numpy()>=h)
  y=np.where((duration<h)&(event>0),event,0)[mask]
  assert np.all(((duration>=h)|(event>0))[mask])
  for name,ps in arrays['test'].items():
   for seed,p in enumerate(ps):
    v=summarize(y,np.ones(len(y)),p[mask,j])
    mature.append(dict(cutoff=cutoff,horizon=h,n=int(mask.sum()),approvals=int((y==1).sum()),model=name,seed=seed+42,**v))
pd.DataFrame(support).to_csv(OUT/'censoring_support.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'calibration_per_seed.csv',index=False)
pd.DataFrame(diagnostics).groupby(['split','horizon','estimator','model']).mean(numeric_only=True).drop(columns='seed').reset_index().to_csv(OUT/'calibration_summary.csv',index=False)
pd.DataFrame(bins).to_csv(OUT/'calibration_bins.csv',index=False)
pd.DataFrame(mature).groupby(['cutoff','horizon','model']).mean(numeric_only=True).drop(columns='seed').reset_index().to_csv(OUT/'mature_unweighted.csv',index=False)
print('TEST DIAGNOSTICS\n',pd.read_csv(OUT/'calibration_summary.csv').query("split=='test' and horizon==720").to_string(index=False),flush=True)
print('SUPPORT\n',pd.DataFrame(support).to_string(index=False),flush=True)
# A censoring-only expansion would impose invented observation times/outcomes;
# report its population size, not an invalid alternative risk estimate.
print('Done; no training, selection, or saved predictions changed.',flush=True)
