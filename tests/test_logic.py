import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from prepare import labels_at,past_features
from model import competing_nll,probabilities,RegReviewModel
from features import Features
from metrics import CensorKM,horizon_targets

def test_future_outcome_is_censored_at_training_cutoff():
    d=pd.DataFrame({'start':pd.to_datetime(['2020-01-01']),'end':pd.to_datetime(['2022-01-01']),'event':[1]})
    duration,event,b,_=labels_at(d,'2020-12-31',{'bin_days':90,'bins':20})
    assert event[0]==0 and duration[0]==365 and b[0]==4

def test_competing_risks_mass_and_monotonicity():
    logits=torch.randn(7,20,4)
    s,c=probabilities(logits)
    assert torch.allclose(s+c.sum(-1),torch.ones_like(s),atol=1e-6)
    assert (s[:,1:]<=s[:,:-1]).all() and (c[:,1:]>=c[:,:-1]).all()

def test_censored_partial_bin_and_event_likelihood():
    logits=torch.zeros(3,3,4,requires_grad=True)
    loss=competing_nll(logits,torch.tensor([0,0,2]),torch.tensor([0,2,1]),reduction='none')
    assert torch.allclose(loss,torch.tensor([0.,2*np.log(4),2*np.log(4)],dtype=torch.float32))
    loss.mean().backward(); assert torch.isfinite(logits.grad).all()

def test_history_same_day_and_future_invariance():
    d=pd.DataFrame({'start':pd.to_datetime(['2020-01-01','2020-01-01','2021-01-01']),'drug_group':['a']*3,'company_group':['b']*3})
    h=past_features(d,730)
    assert np.array_equal(h[:2],np.zeros((2,6)))
    assert np.isclose(h[2,0],np.log(3))
    assert np.array_equal(h[:2],past_features(d.iloc[:2],730))

def test_unknown_category_and_scaler_train_only():
    cfg={'categorical_features':['a'],'text_features':['name']}
    f=Features(cfg,['n']).fit(pd.DataFrame({'a':['x','y'],'name':['甲','乙'],'n':[0.,2.]}))
    n,c,t=f.transform(pd.DataFrame({'a':['z'],'name':['丙'],'n':[100.]}))
    assert c[0,0]==0 and n[0,0]==99 and t[0,0]>0

def test_competing_event_is_known_negative_for_approval():
    km=CensorKM().fit([900,900,900],[1,2,0])
    y,w,_=horizon_targets([90,90,900],[2,0,0],360,km)
    assert y[0]==2 and w[0]>0 and w[1]==0 and w[2]>0

def test_model_forward_and_gradient():
    m=RegReviewModel(2,[3,4],bins=4,dim=32)
    logits,aux=m(torch.randn(3,2),torch.zeros(3,2,dtype=torch.long),torch.zeros(3,64,dtype=torch.long))
    assert logits.shape==(3,4,4) and aux.shape==(3,3,4)
    loss=competing_nll(logits,torch.tensor([1,2,0]),torch.tensor([1,2,3])); loss.backward()
    assert torch.isfinite(loss)
