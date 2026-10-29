"""IPCW horizon scores with explicit censoring support and group-bootstrap CIs."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

class CensorKM:
    def fit(self,duration,event):
        d=np.asarray(duration); e=np.asarray(event)
        self.times=np.unique(d); risk=len(d); g=1.; vals=[]
        for t in self.times:
            at=d==t; n=at.sum(); c=(at&(e==0)).sum()
            g*=1-c/risk; vals.append(g); risk-=n
        self.values=np.asarray(vals); return self
    def at(self,t,left=False):
        ix=np.searchsorted(self.times,np.asarray(t),side='left' if left else 'right')-1
        return np.where(ix<0,1.,self.values[np.maximum(ix,0)])

def horizon_targets(duration,event,horizon,km):
    d=np.asarray(duration); e=np.asarray(event,dtype=int)
    happened=(e>0)&(d<horizon)
    followed=d>=horizon
    y=np.where(happened,e,0)
    g=np.where(happened,km.at(d,left=True),km.at(np.full(len(d),horizon),left=True))
    known=happened|followed
    # Positivity failure is excluded and reported, not hidden by massive weights.
    supported=known&(g>=.05)
    weights=np.where(supported,1/np.maximum(g,.05),0.)
    return y,weights,known&~supported

def score(duration,event,p,h,km):
    y,w,unsupported=horizon_targets(duration,event,h,km); keep=w>0
    answer={'n':len(y),'evaluated':int(keep.sum()),'censoring_support_excluded':int(unsupported.sum()),'horizon_days':h,
            'observed_approvals_before_horizon':int(((y==1)&keep).sum())}
    if not keep.any(): return answer
    # Weighted Brier denominator is total cohort N (standard IPCW definition).
    answer['ipcw_brier_approval']=float(np.sum(w*((y==1)-p[:,1])**2)/len(y))
    answer['ipcw_mass']=float(w.mean())
    yy=(y[keep]==1).astype(int); pp=p[keep,1]; ww=w[keep]
    answer['ipcw_approval_prevalence']=float(np.average(yy,weights=ww))
    if len(np.unique(yy))==2:
        answer['ipcw_auroc_approval']=float(roc_auc_score(yy,pp,sample_weight=ww))
        answer['ipcw_auprc_approval']=float(average_precision_score(yy,pp,sample_weight=ww))
    return answer

def bootstrap(df,p,h,km,n_boot=200,seed=42):
    rng=np.random.default_rng(seed); groups=df.drug_group.fillna('').to_numpy()
    unique=np.unique(groups); index={g:np.flatnonzero(groups==g) for g in unique}; vals=[]
    for _ in range(n_boot):
        ix=np.concatenate([index[g] for g in rng.choice(unique,len(unique),replace=True)])
        s=score(df.duration.to_numpy()[ix],df.target_event.to_numpy()[ix],p[ix],h,km)
        if 'ipcw_auprc_approval' in s: vals.append(s['ipcw_auprc_approval'])
    return {'resamples':len(vals),'auprc_95ci':np.quantile(vals,[.025,.975]).tolist() if vals else None}
