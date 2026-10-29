"""Plot frozen per-seed calibration bins; no model fitting."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True, help='Locally generated calibration_bins.csv; no data are distributed')
p.add_argument('--output',type=Path,default=Path('results/calibration_diagnostics.png'))
args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
names={'matched_logistic':'Matched LR','categorical_hgb':'Tuned HGB','ft_base':'FT baseline','full':'Full model'}

bins=pd.read_csv(args.input)
colors={'matched_logistic':'#a33a2b','categorical_hgb':'#20835d','ft_base':'#d58c13','full':'#285da8'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,h in zip(axes,[360,720]):
 ax.plot([0,1],[0,1],ls='--',color='#888888',lw=1)
 for name in names:
  q=bins[(bins.model==name)&(bins.horizon==h)]
  for seed,g in q.groupby('seed'):
   ax.plot(g.predicted,g.observed,color=colors[name],alpha=1 if seed==42 else .2,lw=1.8 if seed==42 else .7,marker='o' if seed==42 else None,markersize=3,label=names[name] if seed==42 else None)
 ax.set(xlabel='Weighted mean predicted probability',ylabel='Weighted observed approval fraction',title=f'{h} days',xlim=(-.02,1.02),ylim=(-.02,1.02))
 ax.grid(alpha=.15)
axes[0].legend(loc='upper left',frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(args.output,dpi=240);plt.close(fig)
