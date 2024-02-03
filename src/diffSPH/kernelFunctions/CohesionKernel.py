import numpy as np
from diffSPH.kernels import cpow 
import torch


@torch.jit.script
def k(q, dim: int = 2):  
    return torch.where(q < 0.5, 2, 1) * (1-q)**3 * q**3 + torch.where(q < 0.5, -1/64, 0)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return torch.where(q < 0.5, 2, 1) * (-3 *q**2 * (2*q -1) * (1-q)**2)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return torch.where(q < 0.5, 2, 1) * (6 * q * (-5 *q **3 + 10 *q**2 - 6 *q + 1))
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 32 / np.pi
    elif dim == 2: return 32 / np.pi
    else: return 32 / np.pi

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return (((dim - 1) / (rij + 1e-7 * hij)) * dkdq(rij, dim) * hij + d2kdq2(rij, dim)) * C_d(dim) / hij**(dim + 2)
