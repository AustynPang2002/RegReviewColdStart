"""Locked, resumable full-cohort paired-seed ablation study."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
VARIANTS={
    'full':[],
    'ft_base':['--no-text','--no-history','--aux-weight','0'],
    'no_text':['--no-text'],
    'no_history':['--no-history'],
    'no_decay':['--no-decay'],
    'no_multitask':['--aux-weight','0'],
}
LABELS={'full':'完整模型','ft_base':'FT基础版本','no_text':'去药名文本',
        'no_history':'去全部历史','no_decay':'去衰减计数','no_multitask':'去多任务'}
METRICS=['ipcw_auroc_approval','ipcw_auprc_approval','ipcw_brier_approval']

def summarize(out,plan):
    rows=[]
    for seed in plan['seeds']:
        for variant in VARIANTS:
            f=out/f'{variant}_seed{seed}'/'metrics.json'
            if not f.exists():continue
            report=json.loads(f.read_text(encoding='utf-8'))
            for key,score in report['scores'].items():
                split,group,horizon=key.split('/')
                rows.append(dict(variant=variant,seed=seed,split=split,group=group,horizon=int(horizon),
                    best_epoch=min(report['epochs'],key=lambda x:x['val_nll'])['epoch'],**score))
    if not rows:return
    df=pd.DataFrame(rows)
    df.to_csv(out/'per_seed_metrics.csv',index=False,encoding='utf-8-sig')
    aggregate=df.groupby(['variant','split','group','horizon'])[METRICS].agg(['count','mean','std'])
    aggregate.to_csv(out/'mean_std.csv',encoding='utf-8-sig')
    deltas=[]
    for other in VARIANTS:
        if other=='full':continue
        a=df[df.variant.eq('full')]; b=df[df.variant.eq(other)]
        paired=a.merge(b,on=['seed','split','group','horizon'],suffixes=('_full','_other'))
        for keys,g in paired.groupby(['split','group','horizon']):
            for metric in METRICS:
                d=(g[metric+'_full']-g[metric+'_other']).dropna()
                if not len(d):continue
                deltas.append(dict(comparison='full - '+other,split=keys[0],group=keys[1],horizon=keys[2],
                    metric=metric,paired_seeds=len(d),mean_delta=d.mean(),sd_delta=d.std(ddof=1),
                    min_delta=d.min(),max_delta=d.max(),
                    full_better_seeds=int((d<0).sum() if 'brier' in metric else (d>0).sum())))
    pd.DataFrame(deltas).to_csv(out/'paired_seed_differences.csv',index=False,encoding='utf-8-sig')
    complete=df[['variant','seed']].drop_duplicates().shape[0]
    lines=['# 多随机种子消融实验结果','',f'已完成 {complete}/{len(VARIANTS)*len(plan["seeds"])} 次训练。未完成时以下为阶段结果，不得视为最终结论。',
        '', f'每个模型最多{plan["epochs"]}轮、早停耐心5轮；按验证集生存NLL选择检查点。完整训练集，固定时间切分。',
        '五种子均值±样本标准差表示优化随机性，不是独立患者抽样的置信区间。各模型不挑选最佳种子。',
        '', '| 模型 | 测试子组（360天） | 完成种子 | AUROC 均值±SD | AUPRC 均值±SD | Brier 均值±SD |',
        '|---|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for group in ['all','strict_cold_drug','new_drug_application']:
            d=df[df.variant.eq(variant)&df.split.eq('test')&df.group.eq(group)&df.horizon.eq(360)]
            if d.empty:continue
            vals=['{:.4f} ± {:.4f}'.format(d[m].mean(),d[m].std(ddof=1)) if len(d)>1 else '{:.4f}（单次）'.format(d[m].iloc[0]) for m in METRICS]
            lines.append('| '+ ' | '.join([LABELS[variant],group,str(len(d)),*vals])+' |')
    lines+=['','配对差值详见 paired_seed_differences.csv：完整模型减去对应消融版本，AUROC/AUPRC正值较好，Brier负值较好。',
        '5个种子不足以支持强显著性主张；需另外进行药品分组配对bootstrap及独立滚动时间验证。',
        '本轮属于已经查看过测试集后的探索性消融；不能描述为未接触测试数据的确认性实验。',
        'FT基础版本为作者FT骨干加共同竞争风险输出头，并非原论文未经适配的分类模型。',
        '去衰减计数仍保留历史总次数、最近间隔及受理年份；去全部历史仍保留受理年份。',
        '药名冷启动不等于活性成分冷启动；快照随访日期未核实的限制保持不变。']
    (out/'消融验证结果.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps({'completed':complete,'total':len(VARIANTS)*len(plan['seeds']),'updated':time.strftime('%Y-%m-%d %H:%M:%S')},indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--output',default='runs/ablation_server_5seeds');args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Seeds must be unique')
    if min(args.epochs,args.batch_size,args.threads)<1:raise ValueError('Training settings must be positive')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    files=['train.py','features.py','model.py','metrics.py','device_runtime.py','config.json','data/manifest.json','data/cohort.pkl','ablation_study.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    plan={'seeds':args.seeds,'epochs':args.epochs,'batch_size':args.batch_size,'threads':args.threads,'device':args.device,'variants':VARIANTS,'sha256':hashes}
    planfile=out/'study_plan.json'
    if planfile.exists():
        saved=json.loads(planfile.read_text(encoding='utf-8'))
        if not args.summarize_only and saved!=plan:raise RuntimeError('Study inputs/code changed; use a new study directory instead of mixing runs.')
        plan=saved
    else:planfile.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.summarize_only:summarize(out,plan);return
    for seed in plan['seeds']:
        for variant,flags in VARIANTS.items():
            dest=out/f'{variant}_seed{seed}'
            if (dest/'metrics.json').exists():continue
            dest.mkdir(exist_ok=True)
            print(f'START {variant} seed={seed}',flush=True)
            command=[sys.executable,str(ROOT/'train.py'),'--epochs',str(plan['epochs']),'--batch-size',str(plan['batch_size']),
                '--threads',str(plan['threads']),'--seed',str(seed),'--device',plan['device'],'--output',str(dest),*flags]
            env=os.environ.copy();env['PYTHONHASHSEED']=str(seed)
            if plan['device'].startswith('cuda'):env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            with (dest/'training.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode:raise RuntimeError(f'Training failed; inspect {dest / "training.log"}')
            summarize(out,plan)
            print(f'DONE {variant} seed={seed}',flush=True)
    summarize(out,plan)
    print(f'ALL {len(VARIANTS)*len(plan["seeds"])} RUNS COMPLETED',flush=True)

if __name__=='__main__':main()
