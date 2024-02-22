from typing import List, Optional
import torch
from diffSPH.v2.math import mod
from torchCompactRadius import radiusSearch


def neighborSearch(x, y, hx, hy : Optional[torch.Tensor], kernel, dim, periodic : Optional[torch.Tensor] = None, minDomain : Optional[torch.Tensor] = None, maxDomain : Optional[torch.Tensor] = None, mode : str = 'gather', algorithm : str = 'compact'):
    with record_function("NeighborSearch"):
        with record_function("NeighborSearch [adjust Domain]"):
            periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
            if isinstance(periodic, torch.Tensor):
                periodicity = periodic
            # if minDomain is not None and isinstance(minDomain, list):
                # minD = torch.tensor(minDomain).to(x.device).type(x.dtype)
            # else:
            minD = minDomain
            # if maxDomain is not None and isinstance(minDomain, list):
                # maxD = torch.tensor(maxDomain).to(x.device).type(x.dtype)
            # else:
            maxD = maxDomain
        with record_function("NeighborSearch [periodicity]"):
            pos_x = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
            pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
            
        
        with record_function("NeighborSearch [radiusSearch]"):
            if isinstance(hx, float) and (hy is None or isinstance(hy, float)):
                if mode == 'gather':
                    fixedSupport = hx
                elif mode == 'scatter':
                    fixedSupport = hy
                else:
                    fixedSupport = (hx + hy)/2
                i, j = radiusSearch(pos_x, pos_y, support = None,
                                    fixedSupport = torch.tensor(fixedSupport, dtype= torch.float32, device=x.device), 
                                    periodicity = periodicity, domainMin = minD, domainMax = maxD, mode = mode, algorithm = algorithm)
            else:
                i, j = radiusSearch(pos_x, pos_y, 
                                support = (hx if isinstance(hx, torch.Tensor) else torch.ones(x.shape[0]).to(x.device).to(x.dtype) * hx, hy if isinstance(hy, torch.Tensor) else torch.ones(y.shape[0]).to(y.device).to(y.dtype) * hy), fixedSupport = None,
                                periodicity = periodicity, domainMin = minD, domainMax = maxD, mode = mode, algorithm = algorithm)
        
        with record_function("NeighborSearch [compute Support]"):
            if mode == 'gather':
                hij = hx[i] if isinstance(hx, torch.Tensor) else torch.ones(i.shape[0]).to(x.device).to(x.dtype) * hx
            elif mode == 'scatter':
                hij = hy[j] if isinstance(hy, torch.Tensor) else torch.ones(j.shape[0]).to(y.device).to(y.dtype) * hy
            else:
                if isinstance(hx, torch.Tensor) and isinstance(hy, torch.Tensor):
                    hij = (hx[i] + hy[j])/2
                elif isinstance(hx, torch.Tensor):
                    hij = (hx[i] + hy)/2
                elif isinstance(hy, torch.Tensor):
                    hij = (hx + hy[j])/2
                else:
                    hij = torch.ones(j.shape[0]).to(y.device).to(y.dtype) * (hx + hy)/2
            


        with record_function("NeighborSearch [distance Computation]"):
            xij = pos_x[i] - pos_y[j]
            if isinstance(periodic, bool):
                periodicity = [periodic] * dim
            else:
                periodicity = periodic
            xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
            rij = torch.sqrt((xij**2).sum(-1))
            xij = xij / (rij + 1e-7).view(-1,1)
            rij = rij / hij
        with record_function("NeighborSearch [kernel Computation]"):
            Wij = kernel.kernel(rij, hij, dim)
            gradWij = kernel.kernelGradient(rij, xij, hij, dim) 

        return i, j, rij, xij, hij, Wij, gradWij

# @torch.jit.script
# def radiusNaive(x, y, hx, hy, periodic : Optional[List[bool]] = None, minDomain = None, maxDomain = None, mode : str = 'gather'):
#     periodicity = [False] * x.shape[1] if periodic is None else periodic
    
#     pos_x = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
#     pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    
#     distanceMatrices = torch.stack([pos_x[:,i] - pos_y[:,i,None] if not periodic_i else mod(pos_x[:,i] - pos_y[:,i,None], minDomain[i], maxDomain[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
#     distanceMatrix = torch.sqrt(torch.sum(distanceMatrices**2, dim = -1))
    
#     indexI, indexJ = torch.meshgrid(torch.arange(x.shape[0]).to(x.device), torch.arange(y.shape[0]).to(y.device), indexing = 'xy')
#     if mode == 'gather':        
#         gatherMatrix = hx.repeat(hy.shape[0],1)
#         adjacencyDense = distanceMatrix <= gatherMatrix
#         supports = gatherMatrix[adjacencyDense]
#     elif mode == 'scatter':        
#         scatterMatrix = hy.repeat(hx.shape[0],1).mT
#         adjacencyDense = distanceMatrix <= scatterMatrix
#         supports = scatterMatrix[adjacencyDense]
#     else:
#         symmetricMatrix = (hx + hy[:,None]) / 2
#         adjacencyDense = distanceMatrix <= symmetricMatrix
#         supports = symmetricMatrix[adjacencyDense]
    
#     ii = indexI[adjacencyDense]
#     jj = indexJ[adjacencyDense]

#     return ii, jj, distanceMatrix[adjacencyDense], distanceMatrices[adjacencyDense], supports

# # @torch.jit.script
# def neighborSearch(x, y, hx, hy, periodic : Optional[List[bool]] = None, minDomain = None, maxDomain = None, mode : str = 'gather', algorithm : str = 'naive'):
#     periodicity = [False] * x.shape[1] if periodic is None else periodic
    
#     if minDomain is not None and isinstance(minDomain, list):
#         minD = torch.tensor(minDomain).to(x.device).type(x.dtype)
#     else:
#         minD = minDomain
#     if maxDomain is not None and isinstance(minDomain, list):
#         maxD = torch.tensor(maxDomain).to(x.device).type(x.dtype)
#     else:
#         maxD = maxDomain
    
#     i, j, rij, xij, hij = radiusNaive(x, y, 
#                             hx if isinstance(hx, torch.Tensor) else torch.ones(x.shape[0]).to(x.device).to(x.dtype) * hx, hy if isinstance(hy, torch.Tensor) else torch.ones(y.shape[0]).to(y.device).to(y.dtype) * hy, 
#                             periodic = periodicity, minDomain = minD, maxDomain = maxD)
    
#     xij_normed = torch.nn.functional.normalize(xij)
#     rij_normed = rij / hij

#     return i, j, rij_normed, xij_normed, hij


from torch.profiler import record_function
def fluidNeighborSearch(simulationState: dict, config: dict):
    i, j, rij, xij, hij, Wij, gradWij = neighborSearch(simulationState['fluidPositions'], simulationState['fluidPositions'], simulationState['fluidSupports'], simulationState['fluidSupports'], kernel = config['kernel']['function'], dim = config['domain']['dim'], periodic = config['domain']['periodicity'], minDomain = config['domain']['minExtent'], maxDomain = config['domain']['maxExtent'], algorithm = config['neighborhood']['scheme'], mode = config['simulation']['supportScheme'])
    neighborDict = {
        'indices': (i, j),
        'distances': rij,
        'vectors': xij,
        'supports': hij,
        'kernels': Wij,
        'gradients': gradWij,
    }
    return neighborDict


from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('neighborhood', 'scheme', str, 'compact', required = False, export = True),
    ]