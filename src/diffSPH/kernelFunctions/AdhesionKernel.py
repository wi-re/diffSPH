import numpy as np
from diffSPH.kernels import cpow 
import torch


@torch.jit.script
def k(q, dim: int = 2):  
    return torch.where(q > 0.5, ((-4 * q**2 + 6 * q - 2))**(1/4), 0)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return torch.where(q > 0.5, (6 - 8 * q) / (4 * (-4 * q**2 + 6 *q -2)** (3/4)),0)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return torch.where(q > 0.5, (16 *q**2 - 24 *q + 11) / (8 *(-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1)),0)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 0.007
    elif dim == 2: return 0.007
    else: return 0.007

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return (((dim - 1) / (rij + 1e-7 * hij)) * dkdq(rij, dim) * hij + d2kdq2(rij, dim)) * C_d(dim) / hij**(dim + 2)
