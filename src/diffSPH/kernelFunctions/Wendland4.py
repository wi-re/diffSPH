import numpy as np
from diffSPH.kernels import cpow 
import torch


@torch.jit.script
def k(q, dim: int = 2):        
    if dim == 1:
        return cpow(1 - q, 5) * (1 + 5 * q + 8 * q**2)
    else:
        return cpow(1 - q, 6) * (1 + 6 * q + 35/3 * q **2)
@torch.jit.script
def dkdq(q, dim: int = 2):        
    if dim == 1:
        return  -14 * q * (4 *q + 1) * (1 - q)**4
    else:
        return -56/3 * q * (5 * q + 1) * (1 - q)**5
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    if dim == 1:
        return 14 * (24 * q**2 -3 *q -1) * cpow(1-q, 3)
    else:
        return 56/3 * (35 *q**2 -4*q -1) *cpow(1-q,4)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 3/2
    elif dim == 2: return 9 / np.pi
    else: return 495/ (32 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return ((torch.where(rij > -1e-7, ((dim - 1) / (rij *hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij, 0) + d2kdq2(rij, dim)) * C_d(dim)) / hij**(dim + 2)

    # return (torch.where(rij > 1e-7, ((dim - 1) / (rij * hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij + d2kdq2(rij, dim), 0)) * C_d(dim) / hij**(dim + 2)

@torch.jit.script # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def kernelScale(dim: int = 2):
    if dim == 1: return 1.936492
    elif dim == 2: return 2.171239
    else: return 2.207940