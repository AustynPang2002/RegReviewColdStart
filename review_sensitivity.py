"""Locked 2x2 auxiliary-loss sensitivity study; separate from archived results."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
VARIANTS={
 'reference':[],
 'weight_normalized':['--aux-normalization','weights'],
 'supported_consistency':['--consistency-scope','supported'],
 'both_changes':['--aux-normalization','weights','--consistency-scope','supported'],
}

def fingerprint():
    df=pd.read_pickle(ROOT/'data/cohort.pkl')
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    out={'rows':len(df),'splits':df.split.value_counts().to_dict(),'columns':{}}
    for c in cfg['categorical_features']+cfg['text_features']+['受理号','drug_group','company_group','split']:
        raw=json.dumps(df[c].astype(str).tolist(),ensure_ascii=False,separators=(',',':')).encode()
        out['columns'][c]=hashlib.sha256(raw).hexdigest()
    for c in manifest['feature_numeric']+['duration','target_event','time_bin','strict_cold_drug','cold_drug','cold_company']:
        out['columns'][c]=hashlib.sha256(df[c].to_numpy(dtype='<f8').tobytes()).hexdigest()
    out['source_member_sha256']=manifest['member_sha256']
    return out

def summarize(out):
    rows=[]
    for path in sorted(out.glob('*_seed*/metrics.json')):
        r=json.loads(path.read_text(encoding='utf-8')); variant=path.parent.name.rsplit('_seed',1)[0]
        for key,s in r['scores'].items():
            split,group,h=key.split('/')
            rows.append({'variant':variant,'seed':r['arguments']['seed'],'split':split,'group':group,'horizon':int(h),'best_epoch':min(r['epochs'],key=lambda x:x['val_nll'])['epoch'],'epochs':len(r['epochs']),'epoch_seconds':sum(x['seconds'] for x in r['epochs']),**s})
    if rows: pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    return len({(r['variant'],r['seed']) for r in rows})

def main():
    p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS)); p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]);p.add_argument('--device',default='npu:0');p.add_argument('--epochs',type=int,default=15);p.add_argument('--output',default='runs/sensitivity');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    audit=fingerprint(); (out/'data_fingerprint.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.audit_only: print('Data fingerprint saved');return
    source_files=['train.py','features.py','model.py','metrics.py','device_runtime.py','review_losses.py','review_sensitivity.py','config.json']
    plan={'seeds':a.seeds,'variants':{k:VARIANTS[k] for k in a.variants},'epochs':a.epochs,'batch_size':256,'threads':4,'device':a.device,'data_fingerprint':audit,'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},'purpose':'Exploratory post-review 2x2 sensitivity; no test-driven variant selection; original model unchanged','started_utc':None}
    planpath=out/'study_plan.json'
    if planpath.exists():
        old=json.loads(planpath.read_text(encoding='utf-8'))
        if old!=plan: raise RuntimeError('Plan changed; use a new output directory')
    else: planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    lock=out/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time();current=None
    try:
        for seed in a.seeds:
            for variant in a.variants:
                current=f'{variant}_seed{seed}';dest=out/current
                if (dest/'metrics.json').exists(): continue
                dest.mkdir(exist_ok=True)
                status={'state':'running','current':current,'completed':summarize(out),'total':len(a.seeds)*len(a.variants),'elapsed_seconds':time.time()-started}
                (out/'status.json').write_text(json.dumps(status,indent=2))
                print('START',current,flush=True)
                env=os.environ.copy();env['PYTHONHASHSEED']=str(seed)
                cmd=[sys.executable,'-u',str(ROOT/'train.py'),'--device',a.device,'--epochs',str(a.epochs),'--batch-size','256','--threads','4','--seed',str(seed),'--output',str(dest),*VARIANTS[variant]]
                with (dest/'training.log').open('w',encoding='utf-8') as f:
                    r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                if r.returncode: raise RuntimeError(f'{current} exited {r.returncode}; see training.log')
                print('DONE',current,flush=True)
        status={'state':'complete','completed':summarize(out),'total':len(a.seeds)*len(a.variants),'elapsed_seconds':time.time()-started}
        (out/'status.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status),flush=True)
    except BaseException as e:
        (out/'status.json').write_text(json.dumps({'state':'failed','current':current,'error':str(e),'completed':summarize(out),'total':len(a.seeds)*len(a.variants)},indent=2));raise
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__':main()
