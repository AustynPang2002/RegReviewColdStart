from pathlib import Path
import json,pandas as pd,numpy as np
import argparse
parser=argparse.ArgumentParser(description='Compare matched-seed crossover runs, including private predictions.')
parser.add_argument('--input',type=Path,default=Path('results/crossover'))
p=parser.parse_args().input
a=json.loads((p/'card6/study_plan.json').read_text(encoding='utf-8'));b=json.loads((p/'card7/study_plan.json').read_text(encoding='utf-8')); print('Identical sources',a['source_sha256']==b['source_sha256'],'Identical data',a['data_fingerprint']==b['data_fingerprint'])
rows=[]
for card in [6,7]:
 d=pd.read_csv(p/f'card{card}/per_seed_metrics.csv');d=d[(d.split=='test')&(d.group=='all')];d['card']=card;rows.append(d)
d=pd.concat(rows);d.to_csv(p/'metrics_comparison.csv',index=False)
print(d[['card','variant','horizon','best_epoch','epochs','ipcw_auroc_approval','ipcw_auprc_approval','ipcw_brier_approval']].to_string(index=False))
audit=[]
for variant in ['reference','supported_consistency']:
 a=pd.read_csv(p/f'card6/{variant}_seed42/predictions.csv').set_index(['受理号','split']);b=pd.read_csv(p/f'card7/{variant}_seed42/predictions.csv').set_index(['受理号','split']).loc[a.index]; assert np.array_equal(a.target_event,b.target_event) and np.array_equal(a.duration,b.duration)
 for h in [180,360,720]:
  x=(a[f'p_approval_{h}d']-b[f'p_approval_{h}d']).abs()
  audit.append(dict(variant=variant,horizon=h,max_abs_diff=x.max(),mean_abs_diff=x.mean(),exact_equal=bool((x==0).all())))
pd.DataFrame(audit).to_csv(p/'prediction_differences.csv',index=False);print(pd.DataFrame(audit).to_string(index=False))
