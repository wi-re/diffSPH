import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
from torch.profiler import profile, record_function, ProfilerActivity

# Wendland 2 Kernel function and its derivative
@torch.jit.script
def wendland(q, h : float):
    C = 7 / np.pi
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    
# Wendland 2 Kernel function and its derivative
@torch.jit.script
def kernelScalar(q :float, h : float):
    C = 7 / np.pi
    b1 = (1. - q)**4
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    
@torch.jit.script
def wendlandGrad(q,r,h : float):
    C = 7 / np.pi    
    return - r * C / h**3 * (20. * q * (1. -q)**3)[:,None]
# Spiky kernel function used mainly in DFSPH to avoid particle clustering
@torch.jit.script
def spikyGrad(q,r,h : float):
    return -r * 30 / np.pi / h**3 * ((1 - q)**2)[:,None]
# Cohesion kernel is used for the akinci surface tension module
@torch.jit.script
def cohesionKernel(q, h : float):
    res = q.new_zeros(q.shape)
    Cd = -1 / (2 * np.pi) * 1 / h**2
    k1 = 128 * (q-1)**3 * q**3 + 1
    k2 = 64 * (q-1)**3 * q**3
    
    res[q <= 0.5] = k1[q<=0.5]
    res[q > 0.5] = k2[q>0.5]
    
    return -res
# Convenient alias functions for easier usage
@torch.jit.script
def kernel(q , h : float):
    return wendland(q,h)


@torch.jit.script
def kernelGradient(q ,r,h : float):
    return wendlandGrad(q,r,h)

# This function was inteded to be used to swap to different kernel functions
# However, pytorch SPH makes this overly cumbersome so this is not implemented
# TODO: Someday this should be possible in torch script.
def getKernelFunctions(kernel):
    if kernel == 'wendland2':
        return wendland, wendlandGrad



@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p
# @torch.jit.script
# def wendland4(q, h):
#     C = 7 / np.pi
#     b1 = torch.pow(1. - q, 4)
#     b2 = 1.0 + 4.0 * q
#     return b1 * b2 * C / h**2  
    
class Wendland2:
    @staticmethod
    @torch.jit.script
    def kernel(rij, hij, dim : int = 2):
        C = [5/4, 7 / np.pi, 21/ (2 * np.pi)]
        
        if dim == 1:
            k = cpow(1 - rij, 3) * (1 + 3 * rij)
        else:
            k = cpow(1 - rij, 4) * (1 + 4 * rij)
        return k * C[dim - 1] / hij**dim
        
    @staticmethod
    @torch.jit.script
    def kernelGradient(rij, xij, hij, dim : int = 2):
        C = [5/4, 7 / np.pi, 21/ (2 * np.pi)]
        
        if dim == 1:
            k = -12 * rij * cpow(1 - rij, 2)
        else:
            k = -20 * rij * cpow(1 - rij, 3)
        return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
class Wendland4:
    @staticmethod
    @torch.jit.script
    def kernel(rij, hij, dim : int = 2):
        C = [3/2, 9 / np.pi, 495/ (32 * np.pi)]
        
        if dim == 1:
            k = cpow(1 - rij, 5) * (1 + 5 * rij + 8 * rij**2)
        else:
            k = cpow(1 - rij, 6) * (1 + 6 * rij + 35/3 * rij **2)
        return k * C[dim - 1] / hij**dim
        
    @staticmethod
    @torch.jit.script
    def kernelGradient(rij, xij, hij, dim : int = 2):
        C = [3/2, 9 / np.pi, 495/ (32 * np.pi)]
        
        if dim == 1:
            k = -14 * rij * (4 *rij + 1) * (1 - rij)**4
        else:
            k = -56/3 * rij * (5 * rij + 1) * (1 - rij)**5
        return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]        
class Wendland6:
    @staticmethod
    @torch.jit.script
    def kernel(rij, hij, dim : int = 2):
        C = [55/32, 78 / (7 * np.pi), 1365/ (64 * np.pi)]
        
        if dim == 1:
            k = cpow(1 - rij, 7) * (1 + 7 * rij + 19 * rij**2 + 21 * rij**3)
        else:
            k = cpow(1 - rij, 8) * (1 + 8 * rij + 25 * rij**2 + 32 * rij**3)
        return k * C[dim - 1] / hij**dim
        
    @staticmethod
    @torch.jit.script
    def kernelGradient(rij, xij, hij, dim : int = 2):
        C = [55/32, 78 / (7 * np.pi), 1365/ (64 * np.pi)]
        
        if dim == 1:
            k = -6 * rij * (35 * rij**2 + 18 * rij + 3) * (1 - rij)**6
        else:
            k = -22 * rij * (16 * rij**2 + 7 *rij + 1) * (1 - rij)**7
        return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
class CubicSpline:
    @staticmethod
    @torch.jit.script
    def kernel(rij, hij, dim : int = 2):
        C = [8/3, 80 / (7 * np.pi), 16/ (np.pi)]
        k = cpow(1-rij, 3) - 4 * cpow(1/2 - rij,3)
        return k * C[dim - 1] / hij**dim
        
    @staticmethod
    @torch.jit.script
    def kernelGradient(rij, xij, hij, dim : int = 2):
        C = [8/3, 80 / (7 * np.pi), 16/ (np.pi)]
        k = -3 * cpow(1-rij, 2) + 12 * cpow(1/2 - rij,2)
        return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
class QuarticSpline:
    @staticmethod
    @torch.jit.script
    def kernel(rij, hij, dim : int = 2):
        C = [5**5/768, 5**6 * 3 / (2398 * np.pi), 5**6/ (512 * np.pi)]
        k = cpow(1-rij, 4) - 5 * cpow(3/5 - rij, 4) + 10 * cpow(1/5 - rij, 4)
        return k * C[dim - 1] / hij**dim
        
    @staticmethod
    @torch.jit.script
    def kernelGradient(rij, xij, hij, dim : int = 2):
        C = [5**5/768, 5**6 * 3 / (2398 * np.pi), 5**6/ (512 * np.pi)]
        k = -4 * cpow(1-rij, 3) + 20 * cpow(3/5 - rij, 3) - 40 * cpow(1/5 - rij, 3)
        return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
class QuinticSpline:
    @staticmethod
    @torch.jit.script
    def kernel(rij, hij, dim : int = 2):
        C = [3**5/40, 3**7 * 7 / (478 * np.pi), 3**7/ (40 * np.pi)]
        k = cpow(1-rij, 5) - 6 * cpow(2/3 - rij, 5) + 15 * cpow(1/3 - rij, 5)
        return k * C[dim - 1] / hij**dim
        
    @staticmethod
    @torch.jit.script
    def kernelGradient(rij, xij, hij, dim : int = 2):
        C = [3**5/40, 3**7 * 7 / (478 * np.pi), 3**7/ (40 * np.pi)]
        k = -5 * cpow(1-rij, 4) + 30 * cpow(2/3 - rij, 4) - 75 * cpow(1/3 - rij, 4)
        return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
    
def getKernel(kernel: str = 'Wendland2'):
    if kernel == 'Wendland2':
        return Wendland2
    elif kernel == 'Wendland4':
        return Wendland4
    elif kernel == 'Wendland6':
        return Wendland6
    elif kernel == 'CubicSpline':
        return CubicSpline
    elif kernel == 'QuarticSpline':
        return QuarticSpline
    elif kernel == 'QuinticSpline':
        return QuinticSpline
    else: return Wendland2