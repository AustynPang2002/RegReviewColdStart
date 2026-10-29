import hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler

class Features:
    def __init__(self,cfg,numeric,no_history=False):
        self.cats=cfg['categorical_features']; self.numeric=[c for c in numeric if not no_history or c=='entry_year']
        self.text_cols=cfg['text_features']; self.scaler=StandardScaler()
    def fit(self,df):
        self.maps=[{v:i+1 for i,v in enumerate(sorted(set(df[c])-{''}))} for c in self.cats]
        self.imputation_values=df[self.numeric].astype(float).median().fillna(0)
        self.scaler.fit(df[self.numeric].astype(float).fillna(self.imputation_values))
        return self
    @property
    def cardinalities(self): return [len(m)+1 for m in self.maps]
    def transform(self,df):
        # Old archived feature objects retain their original zero-fill behavior.
        num=self.scaler.transform(df[self.numeric].astype(float).fillna(getattr(self,'imputation_values',0))).astype('float32')
        cat=np.array([[m.get(str(v),0) for v in df[c]] for c,m in zip(self.cats,self.maps)],dtype='int64').T
        text=np.zeros((len(df),64),dtype='int64')
        for i,parts in enumerate(df[self.text_cols].itertuples(index=False,name=None)):
            s=' '.join(map(str,parts))[:64]
            for j,ch in enumerate(s): text[i,j]=1+int.from_bytes(hashlib.blake2b(ch.encode(),digest_size=4).digest(),'little')%8191
        return num,cat,text
