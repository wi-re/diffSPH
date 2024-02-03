import numpy as np
from diffSPH.kernels import cpow 
import torch


@torch.jit.script
def k(q, dim: int = 2):        
    if dim == 1:
        return cpow(1 - q, 7) * (1 + 7 * q + 19 * q**2 + 21 * q**3)
    else:
        return cpow(1 - q, 8) * (1 + 8 * q + 25 * q**2 + 32 * q**3)
@torch.jit.script
def dkdq(q, dim: int = 2):        
    if dim == 1:
        return -6 * q * (35 * q**2 + 18 * q + 3) * (1 - q)**6
    else:
        return -22 * q * (16 * q**2 + 7 *q + 1) * (1 - q)**7
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    if dim == 1:
        return 18 * (105 * q**3 + 13 * q**2 - 5 *q -1) * (1-q)**5
    else:
        return 22 * (160 *q**3 + 15 * q**2 - 6*q -1) * (1-q)**6
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 55/32
    elif dim == 2: return 78 / (7 * np.pi)
    else: return 1365/ (64 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return (torch.where(rij > 1e-7, ((dim - 1) / (rij * hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij + d2kdq2(rij, dim), 0)) * C_d(dim) / hij**(dim + 2)