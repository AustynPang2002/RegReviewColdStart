import argparse, copy, json, pickle, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader
from features import Features
from model import RegReviewModel,competing_nll,probabilities
from metrics import CensorKM,horizon_targets,score,bootstrap
from device_runtime import configure,environment
from review_losses import auxiliary_losses

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=20); parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--max-per-split',type=int,default=0); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--output',default='runs/main')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--no-text',action='store_true'); parser.add_argument('--no-history',action='store_true')
    parser.add_argument('--no-decay',action='store_true')
    parser.add_argument('--aux-weight',type=float,default=.2); parser.add_argument('--bootstrap',type=int,default=0)
    parser.add_argument('--aux-normalization',choices=['cohort','weights'],default='cohort')
    parser.add_argument('--consistency-scope',choices=['all','supported','none'],default='all')
    args=parser.parse_args()
    if args.epochs<1: raise ValueError('epochs must be positive')
    configure(args.device,args.seed,args.threads)
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8')); manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    all_df=pd.read_pickle(ROOT/'data/cohort.pkl')
    splits={s:all_df[all_df.split.eq(s)].copy() for s in ['train','val','test']}
    if any(len(d)==0 for d in splits.values()): raise ValueError('Empty time split')
    for s,d in splits.items():
        if args.max_per_split and len(d)>args.max_per_split: splits[s]=d.sample(args.max_per_split,random_state=args.seed).copy()
    numeric=[c for c in manifest['feature_numeric'] if not args.no_decay or 'decayed_count' not in c]
    feats=Features(cfg,numeric,args.no_history).fit(splits['train'])
    km=CensorKM().fit(splits['train'].duration,splits['train'].target_event)
    horizons=[180,360,720]
    loaders={}
    for s,d in splits.items():
        n,c,t=feats.transform(d)
        targets=[]; weights=[]
        for h in horizons:
            yy,ww,_=horizon_targets(d.duration,d.target_event,h,km); targets.append(yy); weights.append(ww)
        tensors=[torch.from_numpy(x) for x in [n,c,t,d.target_event.to_numpy(dtype='int64'),d.time_bin.to_numpy(dtype='int64'),np.array(targets,dtype='int64').T,np.array(weights,dtype='float32').T]]
        generator=torch.Generator().manual_seed(args.seed)
        loaders[s]=DataLoader(TensorDataset(*tensors),batch_size=args.batch_size,shuffle=s=='train',generator=generator)
    model=RegReviewModel(len(feats.numeric),feats.cardinalities,cfg['bins'],no_text=args.no_text).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False)
    best=float('inf'); best_state=None; log=[]; patience=0
    output=ROOT/args.output; output.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        row={'epoch':epoch+1}; start=time.time()
        for s in ['train','val']:
            model.train(s=='train'); total=0.; count=0
            with torch.set_grad_enabled(s=='train'):
                for batch in loaders[s]:
                    n,c,t,e,b,y,w=[x.to(args.device) for x in batch]
                    logits,aux=model(n,c,t)
                    nll=competing_nll(logits,e,b)
                    survival,cif=probabilities(logits)
                    indices=[h//cfg['bin_days']-1 for h in horizons]
                    joint=torch.cat([survival[:,indices,None],cif[:,indices]],-1)
                    aux_loss,consistency=auxiliary_losses(aux,y,w,joint,args.aux_normalization,args.consistency_scope)
                    loss=nll+args.aux_weight*(aux_loss+.1*consistency)
                    if s=='train':
                        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False); opt.step()
                    # Same survival likelihood selects every ablation; auxiliary objective cannot select itself.
                    total+=float(nll.detach())*len(e); count+=len(e)
            row[s+'_nll']=total/count
        row['seconds']=time.time()-start; log.append(row); print(json.dumps(row),flush=True)
        if row['val_nll']<best:
            best=row['val_nll']; best_state=copy.deepcopy(model.state_dict()); patience=0
        else: patience+=1
        if patience>=5: break
    model.load_state_dict(best_state); model.eval()
    scores={}; prediction_rows=[]
    with torch.no_grad():
        for s in ['val','test']:
            all_prob=[]
            for batch in loaders[s]:
                logits,_=model(*[x.to(args.device) for x in batch[:3]])
                sv,ci=probabilities(logits); ix=[h//cfg['bin_days']-1 for h in horizons]
                all_prob.append(torch.cat([sv[:,ix,None],ci[:,ix]],-1).cpu().numpy())
            p=np.concatenate(all_prob); d=splits[s].reset_index(drop=True)
            for group,mask in {'all':np.ones(len(d),bool),'cold_drug':d.cold_drug.to_numpy(),'strict_cold_drug':d.strict_cold_drug.to_numpy(),'cold_company':d.cold_company.to_numpy(),'new_drug_application':d['申请类型（CDE）'].eq('新药').to_numpy()}.items():
                if not mask.any(): continue
                dd=d[mask]; pp=p[mask]
                for j,h in enumerate(horizons):
                    item=score(dd.duration,dd.target_event,pp[:,j],h,km)
                    if args.bootstrap: item.update(bootstrap(dd,pp[:,j],h,km,args.bootstrap,args.seed))
                    scores[f'{s}/{group}/{h}']=item
            pred=d[['受理号','drug_group','start','cold_drug','strict_cold_drug','cold_company','target_event','duration']].copy()
            pred['split']=s
            for j,h in enumerate(horizons):
                for k,label in enumerate(['pending','approval','rejection','withdrawal']): pred[f'p_{label}_{h}d']=p[:,j,k]
            prediction_rows.append(pred)
    cpu_state={k:v.detach().cpu() for k,v in best_state.items()}
    torch.save({'state_dict':cpu_state,'n_numeric':len(feats.numeric),'cardinalities':feats.cardinalities,'bins':cfg['bins'],'no_text':args.no_text},output/'model.pt')
    with (output/'features.pkl').open('wb') as f: pickle.dump(feats,f)
    (output/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'status':'smoke_test' if args.max_per_split else 'experiment','arguments':vars(args),'environment':environment(args.device),'sample_sizes':{s:len(d) for s,d in splits.items()},'epochs':log,'scores':scores,'data_manifest':manifest}
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.concat(prediction_rows).to_csv(output/'predictions.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({'output':str(output),'test_all_360':scores.get('test/all/360')},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
