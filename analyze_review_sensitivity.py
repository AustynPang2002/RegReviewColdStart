"""Summarize completed 2x2 five-seed sensitivity runs; never choose on test."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import CensorKM,horizon_targets
from review_revision import FastMetric

ROOT=Path(__file__).resolve().parent
NAMES=['reference','weight_normalized','supported_consistency','both_changes']
LABELS={'reference':'Fresh reference','weight_normalized':'Weight normalization','supported_consistency':'Supported consistency','both_changes':'Both changes','archived_full':'Archived full model'}
METRICS=['AUROC','AP','Brier'];FIELDS=['ipcw_auroc_approval','ipcw_auprc_approval','ipcw_brier_approval']

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--bootstrap',type=int,default=1000);a=pa.parse_args()
    root=ROOT/'revision_20260925/sensitivity_5seeds';out=root/'analysis';out.mkdir(exist_ok=True)
    df=pd.read_pickle(ROOT/'data/cohort.pkl');test=df[df.split.eq('test')].reset_index(drop=True);ids=test['受理号'].astype(str)
    train=df[df.split.eq('train')];km=CensorKM().fit(train.duration,train.target_event)
    arrays={};perseed=[];reports={};plans=[]
    for worker in [6,7]:
        status=json.loads((root/f'worker{worker}_status.json').read_text())
        if status['state']!='complete' or status['completed']!=10:raise RuntimeError('Do not summarize incomplete five-seed study')
        plans.append(json.loads((root/f'worker{worker}_study_plan.json').read_text(encoding='utf-8')))
    assert plans[0]['source_sha256']==plans[1]['source_sha256'] and plans[0]['data_fingerprint']==plans[1]['data_fingerprint']
    for name in NAMES+['archived_full']:
        predictions=[]
        for seed in range(42,47):
            path=(ROOT/f'results/server01_ablation_5seeds/full_seed{seed}') if name=='archived_full' else root/f'{name}_seed{seed}'
            report=json.loads((path/'metrics.json').read_text(encoding='utf-8'));reports[name,seed]=report
            assert report['sample_sizes']=={'train':69586,'val':4768,'test':9526}
            assert report['arguments']['max_per_split']==0 and report['arguments']['epochs']==15
            pred=pd.read_csv(path/'predictions.csv',dtype={'受理号':str});pred=pred[pred.split.eq('test')].set_index('受理号')
            assert pred.index.is_unique and set(pred.index)==set(ids);pred=pred.loc[ids]
            assert np.allclose(pred.duration,test.duration) and np.array_equal(pred.target_event.to_numpy(),test.target_event.to_numpy())
            predictions.append(pred[[f'p_approval_{h}d' for h in [180,360,720]]].to_numpy())
            for key,s in report['scores'].items():
                split,group,h=key.split('/')
                for metric,field in zip(METRICS,FIELDS):
                    perseed.append(dict(model=name,seed=seed,split=split,group=group,horizon=int(h),metric=metric,value=s.get(field,np.nan),best_epoch=min(report['epochs'],key=lambda x:x['val_nll'])['epoch'],epochs=len(report['epochs']),epoch_seconds=sum(x['seconds'] for x in report['epochs'])))
        arrays[name]=np.stack(predictions)
    pd.DataFrame(perseed).to_csv(out/'per_seed_metrics.csv',index=False)
    summary=pd.DataFrame(perseed).groupby(['model','split','group','horizon','metric']).value.agg(['count','mean','std']).reset_index()
    summary.to_csv(out/'five_seed_summary.csv',index=False)
    rows=[];deltas=[];score_errors=[]
    masks={'all':np.ones(len(test),bool),'strict_cold_drug':test.strict_cold_drug.to_numpy(),'new_drug_application':test['申请类型（CDE）'].eq('新药').to_numpy()}
    contrasts={name+' - reference':{name:1,'reference':-1} for name in NAMES[1:]+['archived_full']}
    contrasts['normalization x masking interaction']={'both_changes':1,'supported_consistency':-1,'weight_normalized':-1,'reference':1}
    for group,mask in masks.items():
        d=test[mask];unique,inverse=np.unique(d.drug_group,return_inverse=True)
        for j,h in enumerate([180,360,720]):
            y,w,_=horizon_targets(d.duration,d.target_event,h,km)
            funcs={name:[FastMetric(y,w,p[mask,j]) for p in ps] for name,ps in arrays.items()}
            points={name:np.mean([fn(np.ones(len(d))) for fn in fns],0) for name,fns in funcs.items()}
            for name in funcs:
                recorded=np.array([np.mean([reports[name,seed]['scores'][f'test/{group}/{h}'][field] for seed in range(42,47)]) for field in FIELDS])
                discrepancy=float(np.max(np.abs(points[name]-recorded)))
                assert discrepancy<1e-7,(name,group,h,discrepancy)
                score_errors.append(discrepancy)
            draws={name:np.empty((a.bootstrap,3)) for name in funcs};rng=np.random.default_rng(20260925)
            for b in range(a.bootstrap):
                count=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))[inverse]
                for name,fns in funcs.items():draws[name][b]=np.mean([fn(count) for fn in fns],0)
            for name in funcs:
                for k,metric in enumerate(METRICS):
                    values=draws[name][:,k];good=np.isfinite(values);lo,hi=np.quantile(values[good],[.025,.975]) if good.any() else [np.nan,np.nan]
                    rows.append(dict(group=group,horizon=h,model=name,metric=metric,estimate=points[name][k],lower=lo,upper=hi,valid_bootstrap=int(good.sum()),n=len(d),clusters=len(unique),positives=int(((y==1)&(w>0)).sum())))
            for label,coefs in contrasts.items():
                for k,metric in enumerate(METRICS):
                    point=sum(points[n][k]*c for n,c in coefs.items());value=sum(draws[n][:,k]*c for n,c in coefs.items());good=np.isfinite(value);lo,hi=np.quantile(value[good],[.025,.975]) if good.any() else [np.nan,np.nan]
                    deltas.append(dict(group=group,horizon=h,comparison=label,metric=metric,difference=point,lower=lo,upper=hi,valid_bootstrap=int(good.sum())))
            print('Bootstrap',group,h,flush=True)
    intervals=pd.DataFrame(rows);differences=pd.DataFrame(deltas)
    intervals.to_csv(out/'cluster_intervals.csv',index=False);differences.to_csv(out/'paired_differences.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,name in zip(axes.flat,NAMES):
        for seed in range(42,47):
            r=reports[name,seed];x=np.array([e['epoch'] for e in r['epochs']]);tr=np.array([e['train_nll'] for e in r['epochs']]);va=np.array([e['val_nll'] for e in r['epochs']]);best=np.argmin(va)
            ax.plot(x,tr,color='#3577a6',alpha=.6,lw=1);ax.plot(x,va,color='#c06a33',alpha=.7,lw=1);ax.scatter(x[best],va[best],color='#c06a33',s=25)
        ax.set(title=LABELS[name],xlabel='Epoch',ylabel='Survival NLL');ax.grid(alpha=.15)
    fig.suptitle('Auxiliary-loss sensitivity: all 20 full-cohort training runs\nBlue: training NLL; orange: validation NLL; points: selected checkpoints',fontsize=14)
    fig.savefig(out/'sensitivity_training.png',dpi=220);fig.savefig(out/'sensitivity_training.svg');plt.close(fig)
    for group in ['all','strict_cold_drug']:
        fig,axes=plt.subplots(3,3,figsize=(16,10),layout='constrained')
        for row,h in enumerate([180,360,720]):
            for col,metric in enumerate(METRICS):
                ax=axes[row,col]
                for i,name in enumerate(NAMES):
                    r=intervals[(intervals.group==group)&(intervals.horizon==h)&(intervals.model==name)&(intervals.metric==metric)].iloc[0]
                    ax.hlines(i,r.lower,r.upper,lw=1.5,color=f'C{i}');ax.scatter(r.estimate,i,color=f'C{i}',s=30)
                ax.set_yticks(range(4),[LABELS[n] for n in NAMES] if col==0 else ['']*4);ax.invert_yaxis();ax.set_title(f'{h} days | {metric}');ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0)
        fig.suptitle(('Overall test' if group=='all' else 'Strict name-cold test')+'\nAuxiliary-loss sensitivity with 95% paired cluster-bootstrap intervals',fontsize=15)
        fig.savefig(out/f'sensitivity_{group}.png',dpi=220);fig.savefig(out/f'sensitivity_{group}.svg');plt.close(fig)
    meta={'completed_training_runs':20,'seeds':[42,43,44,45,46],'design':'2x2 auxiliary normalization x consistency averaging policy','normalization':['cohort mean','weight-sum normalized'],'consistency':['all-entry mean','supported-entry mean'],'primary_model_replaced':False,'test_driven_variant_selection':False,'bootstrap_replicates':a.bootstrap,'bootstrap_estimand':'average per-seed metric conditional on fitted models and training KM','confidence_intervals':'percentile, exploratory, unadjusted for multiplicity','source_and_data_identical_between_workers':True,'hardware':'Ascend910B2; physical cards 6 and 7','score_recomputation_max_abs_error':max(score_errors),'environments':{name:reports[name,42]['environment'] for name in NAMES}}
    (out/'verification.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Completed sensitivity analysis',flush=True)

if __name__=='__main__':main()
