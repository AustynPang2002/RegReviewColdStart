"""Independent, read-only reconciliation against the original Excel archive."""
from pathlib import Path
import io,json,zipfile,hashlib,sys,datetime,collections
import numpy as np,pandas as pd,openpyxl
import argparse
p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
ROOT=args.project_root;OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
df=pd.read_pickle(ROOT/'data/cohort.pkl');test=df[df.split.eq('test')].copy();ids=set(test['受理号'].astype(str))
with zipfile.ZipFile(cfg['archive']) as z:raw=z.read(cfg['member'])
assert hashlib.sha256(raw).hexdigest()==manifest['member_sha256']
wb=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True);ws=wb.active
it=ws.iter_rows(values_only=True);headers=next(it);fields=['受理号','承办日期','审结日期','审评结论','申请事项'];ix=[headers.index(k) for k in fields]
found=[];total=0
for rownum,row in enumerate(it,2):
 total+=1
 if str(row[ix[0]]) in ids:found.append([rownum]+[row[j] for j in ix])
wb.close();assert total==275697
src=pd.DataFrame(found,columns=['excel_row']+fields);assert len(src)==9526 and src['受理号'].is_unique
src=src.set_index('受理号').loc[test['受理号']].reset_index();test=test.reset_index(drop=True)
# Strict ISO parsing independently of prepare.py; original cells are inspected.
formats={}
def parse(x):
 if pd.isna(x) or x=='':return pd.NaT
 formats[type(x).__name__]=formats.get(type(x).__name__,0)+1
 if isinstance(x,(datetime.datetime,datetime.date)):return pd.Timestamp(x)
 return pd.Timestamp(datetime.datetime.strptime(str(x),'%Y-%m-%d'))
start=src['承办日期'].map(parse);end=src['审结日期'].map(parse)
assert start.equals(test.start)
assert ((end==test.end)|(end.isna()&test.end.isna())).all()
codes=src['审评结论'].map(cfg['event_map']).fillna(0).astype(int);assert np.array_equal(codes,test.event)
cutoff=pd.Timestamp('2026-05-19');event_seen=(codes>0)&end.notna()&(end<=cutoff)
duration=(end.where(event_seen,cutoff)-start).dt.days
target=np.where(event_seen&(duration<1800),codes,0)
assert np.array_equal(duration,test.duration) and np.array_equal(target,test.target_event)
assert src['申请事项'].isin(['S：申请生产','S：申请进口']).all()
approved=(codes==1)&event_seen
t=duration[approved];assert len(t)==7275
counts=[]
for h in [180,360,720]:
 rawcount=int((approved&(duration<h)).sum());stored=int(((test.target_event==1)&(test.duration<h)).sum());assert rawcount==stored
 counts.append(dict(horizon=h,source_direct=rawcount,stored_label=stored,exactly_at_horizon=int((approved&(duration==h)).sum())))
edges=[0,180,360,450,540,630,720,np.inf];labels=['0-179','180-359','360-449','450-539','540-629','630-719','720 or more']
hist=[]
for lo,hi,label in zip(edges[:-1],edges[1:],labels):
 m=approved&(duration>=lo)&(duration<hi)
 hist.append(dict(days=label,n=int(m.sum()),percent_of_recorded_approvals=float(m.sum()/approved.sum()*100),accepted_2023=int((m&(start.dt.year==2023)).sum()),accepted_2024=int((m&(start.dt.year==2024)).sum())))
pd.DataFrame(hist).to_csv(OUT/'approval_time_distribution.csv',index=False)
pd.DataFrame(counts).to_csv(OUT/'approval_horizon_reconciliation.csv',index=False)
pd.DataFrame({'days':t.value_counts().index,'n':t.value_counts().values}).sort_values('days').to_csv(OUT/'approval_times_by_day.csv',index=False)
end[approved].value_counts().rename_axis('completion_date').reset_index(name='n').sort_values('completion_date').to_csv(OUT/'approval_completion_dates.csv',index=False)
mapping=pd.crosstab(src['审评结论'],pd.Series(target,name='target_event'));mapping.to_csv(OUT/'outcome_mapping_test.csv')
quantiles={str(q):float(t.quantile(q)) for q in [0,.1,.25,.5,.75,.9,1]}
audit={'source_member_sha256':hashlib.sha256(raw).hexdigest(),'source_hash_matches_original_manifest':True,'source_records':total,'test_records':len(test),'test_ids_unique_in_source':True,'all_test_start_end_and_event_codes_match_source':True,'all_test_durations_and_target_events_reproduced':True,'date_cell_types':formats,'date_parser':'datetime.strptime %Y-%m-%d; native datetime cells accepted; no date imputation for observed approvals','observed_approvals':int(approved.sum()),'raw_cause1_count':int((codes==1).sum()),'approval_duration_quantiles':quantiles,'boundary_counts':counts,'approval_times_360_to_719':int(((t>=360)&(t<720)).sum()),'top_durations':{str(k):int(v) for k,v in t.value_counts().head(10).items()},'largest_completion_date_count':int(end[approved].value_counts().max()),'unique_approval_completion_dates':int(end[approved].nunique()),'processing_mismatch_found':False,'scope':'Record consistency only, not independent verification of source date semantics, posting time or continuous follow-up.'}
(OUT/'approval_time_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2));print(pd.DataFrame(hist).to_string(index=False))
