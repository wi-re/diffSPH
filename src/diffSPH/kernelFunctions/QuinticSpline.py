import numpy as np
from diffSPH.kernels import cpow 
import torch


@torch.jit.script
def k(q, dim: int = 2):        
    return cpow(1-q, 5) - 6 * cpow(2/3 - q, 5) + 15 * cpow(1/3 - q, 5)
@torch.jit.script
def dkdq(q, dim: int = 2):     
    return -5 * cpow(1-q, 4) + 30 * cpow(2/3 - q, 4) - 75 * cpow(1/3 - q, 4)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return 20 * cpow(1-q,3) - 120 * cpow(2/3 - q, 3) + 300 * cpow(1/3 - q, 3)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 3**5/40
    elif dim == 2: return 3**7 * 7 / (478 * np.pi)
    else: return 3**7/ (40 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return (torch.where(rij > 1e-7, ((dim - 1) / (rij * hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij + d2kdq2(rij, dim), 0)) * C_d(dim) / hij**(dim + 2)