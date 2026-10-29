"""Explicit sensitivity controls; defaults reproduce the archived objective."""
import torch
from torch.nn import functional as F

def auxiliary_losses(aux, target, weights, joint, normalization='cohort', consistency_scope='all'):
    ce=F.cross_entropy(aux.reshape(-1,4),target.reshape(-1),reduction='none').reshape_as(weights)
    if normalization not in ('cohort','weights'): raise ValueError(normalization)
    if consistency_scope not in ('all','supported','none'): raise ValueError(consistency_scope)
    denominator=weights.numel() if normalization=='cohort' else weights.sum().clamp_min(1e-12)
    supervised=(ce*weights).sum()/denominator
    kl=F.kl_div(aux.log_softmax(-1),joint.detach(),reduction='none').sum(-1)
    if consistency_scope=='all': consistency=kl.mean()
    elif consistency_scope=='supported':
        mask=weights.gt(0).to(kl.dtype)
        consistency=(kl*mask).sum()/mask.sum().clamp_min(1)
    else: consistency=kl.sum()*0
    return supervised,consistency
