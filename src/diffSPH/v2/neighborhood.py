from typing import List, Optional
import torch
from diffSPH.v2.math import mod

@torch.jit.script
def radiusNaive(x, y, hx, hy, periodic : Optional[List[bool]] = None, minDomain = None, maxDomain = None, mode : str = 'gather'):
    periodicity = [False] * x.shape[1] if periodic is None else periodic
    
    pos_x = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    
    distanceMatrices = torch.stack([pos_x[:,i] - pos_y[:,i,None] if not periodic_i else mod(pos_x[:,i] - pos_y[:,i,None], minDomain[i], maxDomain[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
    distanceMatrix = torch.sqrt(torch.sum(distanceMatrices**2, dim = -1))
    
    indexI, indexJ = torch.meshgrid(torch.arange(x.shape[0]).to(x.device), torch.arange(y.shape[0]).to(y.device), indexing = 'xy')
    if mode == 'gather':        
        gatherMatrix = hx.repeat(hy.shape[0],1)
        adjacencyDense = distanceMatrix <= gatherMatrix
        supports = gatherMatrix[adjacencyDense]
    elif mode == 'scatter':        
        scatterMatrix = hy.repeat(hx.shape[0],1).mT
        adjacencyDense = distanceMatrix <= scatterMatrix
        supports = scatterMatrix[adjacencyDense]
    else:
        symmetricMatrix = (hx + hy[:,None]) / 2
        adjacencyDense = distanceMatrix <= symmetricMatrix
        supports = symmetricMatrix[adjacencyDense]
    
    ii = indexI[adjacencyDense]
    jj = indexJ[adjacencyDense]

    return ii, jj, distanceMatrix[adjacencyDense], distanceMatrices[adjacencyDense], supports

@torch.jit.script
def neighborSearch(x, y, hx, hy, periodic : Optional[List[bool]] = None, minDomain = None, maxDomain = None, mode : str = 'gather', algorithm : str = 'naive'):
    periodicity = [False] * x.shape[1] if periodic is None else periodic
    
    if minDomain is not None and isinstance(minDomain, list):
        minD = torch.tensor(minDomain).to(x.device).type(x.dtype)
    else:
        minD = minDomain
    if maxDomain is not None and isinstance(minDomain, list):
        maxD = torch.tensor(maxDomain).to(x.device).type(x.dtype)
    else:
        maxD = maxDomain
    
    i, j, rij, xij, hij = radiusNaive(x, y, 
                            hx if isinstance(hx, torch.Tensor) else torch.ones(x.shape[0]).to(x.device).to(x.dtype) * hx, hy if isinstance(hy, torch.Tensor) else torch.ones(y.shape[0]).to(y.device).to(y.dtype) * hy, 
                            periodic = periodicity, minDomain = minD, maxDomain = maxD)
    
    xij_normed = torch.nn.functional.normalize(xij)
    rij_normed = rij / hij

    return i, j, rij_normed, xij_normed, hij