import io, json, zipfile
from pathlib import Path
import pandas as pd

def main():
    root=Path(__file__).parent
    (root/'data').mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(cfg['archive']) as z:
        df=pd.read_excel(io.BytesIO(z.read(cfg['member'])),dtype=str).fillna('')
    df.to_pickle(root/'data/raw_applications.pkl')
    cols=['审评结论','申请类型（CDE）','申请事项','注册分类（CDE）','创新类型']
    report={c:df[c].value_counts().head(80).to_dict() for c in cols}
    report['rows']=len(df)
    report['dates']={c:df[c].head(5).tolist() for c in ['承办日期','审结日期']}
    (root/'data/audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
