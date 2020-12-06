import os
import torch
from torch.autograd import Variable

FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    if not volatile:
        with torch.no_grad():
            x = torch.from_numpy(ndarray)
    else:
        x = torch.from_numpy(ndarray)
    x.requires_grad_(requires_grad)
    return x.type(dtype)

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()