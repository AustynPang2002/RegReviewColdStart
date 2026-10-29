"""Official FT-Transformer backbone + inductive text + coherent competing hazards.
FT backbone is imported unchanged from the Apache-2.0 author implementation.
The competing-hazard head below is a new implementation, NOT a DeepHit reproduction.
"""
import torch
from torch import nn
from rtdl_revisiting_models import FTTransformer

class RegReviewModel(nn.Module):
    def __init__(self,n_numeric,cardinalities,bins=20,dim=64,no_text=False,dropout=.15):
        super().__init__(); self.bins=bins; self.no_text=no_text
        self.backbone=FTTransformer(n_cont_features=n_numeric,cat_cardinalities=cardinalities,
            d_out=dim,n_blocks=2,d_block=dim,attention_n_heads=4,attention_dropout=dropout,
            ffn_d_hidden=None,ffn_d_hidden_multiplier=4/3,ffn_dropout=dropout,residual_dropout=0.)
        self.text=nn.Embedding(8192,dim,padding_idx=0)
        self.gate=nn.Linear(dim*2,dim)
        self.fusion=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Dropout(dropout))
        self.hazards=nn.Linear(dim,bins*4)
        self.horizon_head=nn.Linear(dim,3*4)
    def forward(self,num,cat,text):
        z=self.backbone(num,cat)
        if not self.no_text:
            mask=text.ne(0).unsqueeze(-1)
            t=(self.text(text)*mask).sum(1)/mask.sum(1).clamp_min(1)
            if self.training:
                t=t*(torch.rand((len(t),1),device=t.device)>.15)
            z=z+torch.sigmoid(self.gate(torch.cat([z,t],-1)))*t
        z=self.fusion(z)
        return self.hazards(z).reshape(-1,self.bins,4), self.horizon_head(z).reshape(-1,3,4)

def competing_nll(logits,event,time_bin,reduction='mean'):
    """Cause=0 means censoring; censored b observes [0,b), event b observes [0,b]."""
    lp=logits.log_softmax(-1); T=lp.shape[1]
    previous=torch.arange(T,device=lp.device)[None,:]<time_bin[:,None]
    loss=-(lp[:,:,0]*previous).sum(1)
    observed=event.gt(0)&time_bin.lt(T)
    # Fixed-shape gather avoids dynamic nonzero/indexed writes on accelerators.
    selected=lp.gather(1,time_bin.clamp(0,T-1)[:,None,None].expand(-1,1,4)).squeeze(1)
    event_lp=selected.gather(1,event[:,None]).squeeze(1)
    loss=loss-event_lp*observed.to(lp.dtype)
    return loss.mean() if reduction=='mean' else loss

def probabilities(logits):
    hazard=logits.softmax(-1)
    survival=hazard[:,:,0].cumprod(1)
    before=torch.cat([torch.ones_like(survival[:,:1]),survival[:,:-1]],1)
    cif=(before[:,:,None]*hazard[:,:,1:]).cumsum(1)
    return survival,cif
