import numpy as np
import pandas as pd
import torch
from features import Features
from review_losses import auxiliary_losses

def test_imputation_is_training_median_and_legacy_compatible():
    cfg={'categorical_features':['a'],'text_features':['name']}
    train=pd.DataFrame({'a':['a']*3,'name':['x']*3,'n':[2.,4.,np.nan]})
    f=Features(cfg,['n']).fit(train)
    test=pd.DataFrame({'a':['b'],'name':['y'],'n':[np.nan]})
    assert f.imputation_values['n']==3 and f.transform(test)[0][0,0]==0
    del f.imputation_values
    assert np.isfinite(f.transform(test)[0]).all()

def test_supported_consistency_excludes_unsupported_gradients():
    aux=torch.randn(2,3,4,requires_grad=True)
    joint=torch.softmax(torch.randn(2,3,4),-1).requires_grad_()
    w=torch.tensor([[1.,0.,1.],[0.,0.,1.]])
    y=torch.zeros(2,3,dtype=torch.long)
    a,c=auxiliary_losses(aux,y,w,joint,'weights','supported')
    (a+c).backward()
    assert torch.all(aux.grad[w==0]==0) and joint.grad is None
    a,c=auxiliary_losses(aux,y,torch.zeros_like(w),joint,'weights','supported')
    assert a==0 and c==0

def test_default_loss_reproduces_archived_definition():
    from torch.nn import functional as F
    aux=torch.randn(2,3,4); y=torch.zeros(2,3,dtype=torch.long); w=torch.rand(2,3)
    joint=torch.softmax(torch.randn(2,3,4),-1)
    a,c=auxiliary_losses(aux,y,w,joint)
    assert torch.allclose(a,(F.cross_entropy(aux.reshape(-1,4),y.reshape(-1),reduction='none').reshape_as(w)*w).mean())
    assert torch.allclose(c,F.kl_div(aux.log_softmax(-1),joint,reduction='batchmean')/3)

def test_fast_bootstrap_scores_match_expanded_sample_with_ties():
    from review_revision import FastMetric
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.array([0,1,1,0,2]); w=np.array([1.,2.,0.,3.,1.]); p=np.array([.2,.5,.5,.5,.1])
    count=np.array([2,1,3,2,1]); ix=np.repeat(np.arange(5),count)
    a=FastMetric(y,w,p)(count)
    assert np.isclose(a[0],roc_auc_score(y[ix]==1,p[ix],sample_weight=w[ix]))
    assert np.isclose(a[1],average_precision_score(y[ix]==1,p[ix],sample_weight=w[ix]))
    assert np.isclose(a[2],np.sum(w[ix]*((y[ix]==1)-p[ix])**2)/len(ix))

def test_name_hash_matches_neural_bag():
    from review_revision import text_matrix
    cfg={'categorical_features':['a'],'text_features':['name']}
    d=pd.DataFrame({'a':['a'],'name':['ababa'],'n':[0.]})
    f=Features(cfg,['n']).fit(d); tokens=f.transform(d)[2][0]; tokens=tokens[tokens>0]
    expected=np.bincount(tokens,minlength=8192)/len(tokens)
    assert np.allclose(text_matrix(d,['name']).toarray()[0],expected)
