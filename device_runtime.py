"""Explicit CPU/CUDA/Ascend NPU runtime, with no silent device fallback."""
import importlib.metadata
import os
import platform
import random
import numpy as np
import torch

def configure(device='cpu',seed=42,threads=4):
    backend=device.split(':')[0]
    if backend not in {'cpu','cuda','npu'}:raise ValueError('Use cpu, cuda:N or npu:N')
    if backend=='npu':
        try:import torch_npu  # noqa: F401; registers the npu backend
        except ImportError as exc:raise RuntimeError('Install matching torch/torch_npu/CANN before selecting NPU') from exc
        if not torch.npu.is_available():raise RuntimeError('NPU unavailable; check driver, CANN environment and device permissions')
        torch.npu.set_device(device)
        torch.npu.manual_seed_all(seed)
    elif backend=='cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(device);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return torch.device(device)

def environment(device):
    packages={}
    for name in ['torch','torch-npu','numpy','pandas','scikit-learn','rtdl_revisiting_models']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    info={'device':str(device),'python':platform.python_version(),'platform':platform.platform(),
          'packages':packages,'deterministic_requested':torch.are_deterministic_algorithms_enabled()}
    if str(device).startswith('npu'):info['device_name']=torch.npu.get_device_name(device)
    elif str(device).startswith('cuda'):info['device_name']=torch.cuda.get_device_name(device)
    return info
