from typing import List, Optional
import torch
from diffSPH.v2.math import mod
from torch.profiler import record_function


try:
    from torchCompactRadius import radiusSearch
    hasClusterRadius = True
except ModuleNotFoundError:
    from diffSPH.v2.neighborhoodFallback.neighborhood import radiusSearch
    hasClusterRadius = False
    # pass


def neighborSearch(x, y, hx, hy : Optional[torch.Tensor], kernel, dim, periodic : Optional[torch.Tensor] = None, minDomain : Optional[torch.Tensor] = None, maxDomain : Optional[torch.Tensor] = None, mode : str = 'gather', algorithm : str = 'compact' if hasClusterRadius else 'naive'):
    with record_function("NeighborSearch"):
        with record_function("NeighborSearch [adjust Domain]"):
            periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
            if isinstance(periodic, torch.Tensor):
                periodicity = periodic
            if isinstance(periodic, bool):
                periodicity = torch.tensor([periodic] * x.shape[1], dtype = torch.bool).to(x.device)
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
            # rij = torch.sqrt((xij**2).sum(-1))
            rij = torch.linalg.norm(xij, dim = -1)
            xij = xij / (rij + 1e-7).view(-1,1)
            rij = rij / hij
        with record_function("NeighborSearch [kernel Computation]"):
            Wij = kernel.kernel(rij, hij, dim)
            gradWij = kernel.kernelGradient(rij, xij, hij, dim) 

        return i, j, rij, xij, hij, Wij, gradWij


def neighborSearchVerlet(x, y, hx, hy : Optional[torch.Tensor], kernel, dim, periodic : Optional[torch.Tensor] = None, minDomain : Optional[torch.Tensor] = None, maxDomain : Optional[torch.Tensor] = None, mode : str = 'gather', algorithm : str = 'compact' if hasClusterRadius else 'naive'):
    with record_function("NeighborSearch"):
        with record_function("NeighborSearch [adjust Domain]"):
            periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
            if isinstance(periodic, torch.Tensor):
                periodicity = periodic
            if isinstance(periodic, bool):
                periodicity = torch.tensor([periodic] * x.shape[1], dtype = torch.bool).to(x.device)
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
            
        with torch.no_grad():
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
            # rij = torch.sqrt((xij**2).sum(-1))
            rij = torch.linalg.norm(xij, dim = -1)
            xij = xij / (rij + 1e-7).view(-1,1)
            rij = rij / hij
            # rij = torch.clamp(rij, 0, 1)
        with record_function("NeighborSearch [kernel Computation]"):
            Wij = kernel.kernel(rij, hij, dim)
            gradWij = kernel.kernelGradient(rij, xij, hij, dim) 

        return i, j, hij


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

from diffSPH.v2.util import countUniqueEntries
from torch.profiler import record_function
# def neighborSearch(stateA: dict, stateB : dict, config: dict, computeKernels = True, priorNeighborhood = None):
#     if priorNeighborhood is not None:
#         # print('Using prior neighborhood')
#         return evalNeighborhood(*priorNeighborhood['fullIndices'], priorNeighborhood['fullSupports'], stateA, stateB, config, computeKernels = computeKernels, priorNeighborhood = priorNeighborhood)
    
#     i, j, hij = neighborSearchVerlet(stateA['positions'], stateB['positions'], stateA['supports'] * config['neighborhood']['verletScale'], stateB['supports'] * config['neighborhood']['verletScale'], kernel = config['kernel']['function'], dim = config['domain']['dim'], periodic = config['domain']['periodicity'], minDomain = config['domain']['minExtent'], maxDomain = config['domain']['maxExtent'], algorithm = config['neighborhood']['scheme'], mode = config['simulation']['supportScheme'])

#     neighborDict = evalNeighborhood(i, j, hij, stateA, stateB, config, computeKernels = computeKernels)
#     # neighborDict['initialPositions'] = simulationState['fluidPositions'].clone()
#     return neighborDict

def evalNeighborhood(i, j, hij, stateA, stateB : dict, config: dict, computeKernels = True, priorNeighborhood: Optional[dict] = None):
    periodic = config['domain']['periodicity']
    minDomain = config['domain']['minExtent']
    maxDomain = config['domain']['maxExtent']

    x = stateA['positions']
    y = stateB['positions']

    # initialPositions = simulationState['fluidPositions'].clone()
    initialPositions = (stateA['positions'].clone(), stateB['positions'].clone())

    if priorNeighborhood is not None:
        if 'initialPositions' in priorNeighborhood:
            if \
                (priorNeighborhood['initialPositions'][0].shape == stateA['positions'].shape) and\
                (priorNeighborhood['initialPositions'][1].shape == stateB['positions'].shape):
                initialPositions = priorNeighborhood['initialPositions']
                distancesA = torch.linalg.norm(initialPositions[0] - stateA['positions'], dim = -1)
                distancesB = torch.linalg.norm(initialPositions[1] - stateB['positions'], dim = -1)
                maxDistance = max(distancesA.max(), distancesB.max())
                minSupport = min(stateA['supports'].min(), stateB['supports'].min())
                if maxDistance * 2 > (config['neighborhood']['verletScale'] - 1) * minSupport:    

                    # print('Recomputing neighborhood (maxDistance = ', maxDistance / minSupport, ')')
                    i, j, hij = neighborSearchVerlet(stateA['positions'], stateB['positions'], stateA['supports'] * config['neighborhood']['verletScale'], stateB['supports'] * config['neighborhood']['verletScale'], kernel = config['kernel']['function'], dim = config['domain']['dim'], periodic = config['domain']['periodicity'], minDomain = config['domain']['minExtent'], maxDomain = config['domain']['maxExtent'], algorithm = config['neighborhood']['scheme'], mode = config['simulation']['supportScheme'])
                    hij / config['neighborhood']['verletScale']
                    initialPositions = (stateA['positions'].clone(), stateB['positions'].clone())
                # else:
                    # print('Reusing prior neighborsearch (maxDistance = ', maxDistance / minSupport, ')')
            else:
                i, j, hij = neighborSearchVerlet(stateA['positions'], stateB['positions'], stateA['supports'] * config['neighborhood']['verletScale'], stateB['supports'] * config['neighborhood']['verletScale'], kernel = config['kernel']['function'], dim = config['domain']['dim'], periodic = config['domain']['periodicity'], minDomain = config['domain']['minExtent'], maxDomain = config['domain']['maxExtent'], algorithm = config['neighborhood']['scheme'], mode = config['simulation']['supportScheme'])
                hij / config['neighborhood']['verletScale']
                initialPositions = (stateA['positions'].clone(), stateB['positions'].clone())
            # else:
                # print('Recomputing neighborsearch (invalid configuration)')
        else:
            # print('Recomputing neighborhood (no prior positions)')
            i, j, hij = neighborSearchVerlet(stateA['positions'], stateB['positions'], stateA['supports'] * config['neighborhood']['verletScale'], stateB['supports'] * config['neighborhood']['verletScale'], kernel = config['kernel']['function'], dim = config['domain']['dim'], periodic = config['domain']['periodicity'], minDomain = config['domain']['minExtent'], maxDomain = config['domain']['maxExtent'], algorithm = config['neighborhood']['scheme'], mode = config['simulation']['supportScheme'])
            hij / config['neighborhood']['verletScale']
    # else:
        # print('Recomputing neighborhood (no prior neighborhood)')
    # hij = hij 
    actual_hij = hij / config['neighborhood']['verletScale']

    # (i,j) = simulationState['fluidNeighborhood']['indices']
    # hij = simulationState['fluidSupports'] 
    dim = config['domain']['dim']
    kernel = config['kernel']['function']

    with record_function("NeighborSearch [adjust Domain]"):
        periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
        if isinstance(periodic, torch.Tensor):
            periodicity = periodic
        if isinstance(periodic, bool):
            periodicity = torch.tensor([periodic] * x.shape[1], dtype = torch.bool).to(x.device)
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

    with record_function("NeighborSearch [distance Computation]"):
        xij = pos_x[i] - pos_y[j]
        if isinstance(periodic, bool):
            periodicity = [periodic] * dim
        else:
            periodicity = periodic
        xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
        # rij = torch.sqrt((xij**2).sum(-1))
        rij = torch.linalg.norm(xij, dim = -1)
        xij = xij / (rij + 1e-7).view(-1,1)
        rij = rij / actual_hij

        mask = rij <= 1.0
        rij = rij[mask]
        xij = xij[mask,:]
        iFiltered = i[mask]
        jFiltered = j[mask]
        hijFiltered = actual_hij[mask]

    numNeighbors = countUniqueEntries(iFiltered, pos_x)[1].to(torch.int32)
    neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors.device), torch.cumsum(numNeighbors, dim = 0).to(torch.int32)))[:-1]

        # rij = torch.clamp(rij, 0, 1)
    if computeKernels:
        with record_function("NeighborSearch [kernel Computation]"):
            Wij = kernel.kernel(rij, hijFiltered, dim)
            gradWij = kernel.kernelGradient(rij, xij, hijFiltered, dim) 

        neighborDict = {
            'indices': (iFiltered, jFiltered),
            'fullIndices': (i, j),
            'distances': rij,
            'vectors': xij,
            'supports': hijFiltered,
            'fullSupports': hij,
            'kernels': Wij,
            'gradients': gradWij,
            'initialPositions': initialPositions,
            'numNeighbors': numNeighbors,
            'neighborOffsets': neighborOffsets,
        }
    else:
        neighborDict = {
            'indices': (iFiltered, jFiltered),
            'fullIndices': (i, j),
            'distances': rij,
            'vectors': xij,
            'supports': hijFiltered,
            'fullSupports': hij,
            'initialPositions': initialPositions,
            'numNeighbors': numNeighbors,
            'neighborOffsets': neighborOffsets,
        }
    return neighborDict

from diffSPH.v2.math import mod
# mode = 'scatter'

from torchCompactRadius.neighborhood import neighborSearch, buildDataStructure, neighborSearchExisting, radiusSearch

def searchNeighbors(state, config):    
    # print('fluid - fluid neighbor search')
    state['fluid']['datastructure'], state['fluid']['neighborhood'] = neighborSearch(state['fluid'], state['fluid'], config, 
        computeKernels = True, 
        priorState = None if 'neighborhood' not in state['fluid'] else state['fluid']['neighborhood'],
        neighborDatastructure = None if 'datastructure' not in state['fluid'] else state['fluid']['datastructure'],
        verbose = False)
    state['fluid']['numNeighbors'] = state['fluid']['neighborhood']['numNeighbors']
    # _, state['fluid']['numNeighbors'] = countUniqueEntries(state['fluid']['neighborhood']['indices'][0], state['fluid']['positions'])
    # torch.cuda.synchronize()
    if config['boundary']['active']:
        # print('boundary - boundary neighbor search')
        state['boundary']['datastructure'], state['boundary']['neighborhood'] = neighborSearch(state['boundary'], state['boundary'], config, 
            computeKernels = True, 
            priorState = None if 'neighborhood' not in state['boundary'] else state['boundary']['neighborhood'],
            neighborDatastructure = None if 'datastructure' not in state['boundary'] else state['fluid']['datastructure'],
            verbose = False)
        state['boundary']['numNeighbors'] = state['boundary']['neighborhood']['numNeighbors']
        _, state['boundary']['numNeighbors'] = countUniqueEntries(state['boundary']['neighborhood']['indices'][0], state['boundary']['positions'])
    
    # torch.cuda.synchronize()
    if config['boundary']['active']:
        # print('boundary - fluid neighbor search')
        _, state['fluidToBoundaryNeighborhood'] = neighborSearch(state['boundary'], state['fluid'], config, True, 
            None if 'fluidToBoundaryNeighborhood' not in state else state['fluidToBoundaryNeighborhood'], 
            state['fluid']['datastructure'], False)
        # print('fluid - boundary neighbor search')
        _, state['boundaryToFluidNeighborhood'] = neighborSearch(state['fluid'], state['boundary'], config, True, None if 'boundaryToFluidNeighborhood' not in state else state['boundaryToFluidNeighborhood'], state['boundary']['datastructure'], False)
        # print('boundary - fluid neighbor search count')
        # state['boundaryToFluidNeighborhood'] = neighborSearch(state['fluid'], state['boundary'], config, priorNeighborhood = None if 'boundaryToFluidNeighborhood' not in state else state['boundaryToFluidNeighborhood'])
        state['boundaryToFluidNumNeighbors'] = state['boundaryToFluidNeighborhood']['numNeighbors']
        # _, state['boundaryToFluidNumNeighbors'] = countUniqueEntries(state['boundaryToFluidNeighborhood']['indices'][0], state['fluid']['positions'])


        # print('fluid - boundary neighbor search couunt')
        # state['fluidToBoundaryNeighborhood'] = neighborSearch(state['boundary'], state['fluid'], config, priorNeighborhood = None if 'fluidToBoundaryNeighborhood' not in state else state['fluidToBoundaryNeighborhood'])
        state['fluidToBoundaryNumNeighbors'] = state['fluidToBoundaryNeighborhood']['numNeighbors']
        # _, state['fluidToBoundaryNumNeighbors'] = countUniqueEntries(state['fluidToBoundaryNeighborhood']['indices'][0], state['boundary']['positions'])

    # if config['boundary']['active']:
    #     # print('boundary - fluid neighbor search')
    #     _, state['boundaryToFluidNeighborhood'] = neighborSearch(state['boundary'], state['fluid'], config, True, None, state['fluid']['datastructure'], True)
    #     # print('fluid - boundary neighbor search')
    #     _, state['fluidToBoundaryNeighborhood'] = neighborSearch(state['fluid'], state['boundary'], config, True, None, state['boundary']['datastructure'], True)
    #     # state['boundaryToFluidNeighborhood'] = neighborSearch(state['fluid'], state['boundary'], config, priorNeighborhood = None if 'boundaryToFluidNeighborhood' not in state else state['boundaryToFluidNeighborhood'])
    #     _, state['boundaryToFluidNumNeighbors'] = countUniqueEntries(state['boundaryToFluidNeighborhood']['indices'][0], state['fluid']['positions'])
    #     # state['fluidToBoundaryNeighborhood'] = neighborSearch(state['boundary'], state['fluid'], config, priorNeighborhood = None if 'fluidToBoundaryNeighborhood' not in state else state['fluidToBoundaryNeighborhood'])
    #     _, state['fluidToBoundaryNumNeighbors'] = countUniqueEntries(state['fluidToBoundaryNeighborhood']['indices'][0], state['boundary']['positions'])


def getNeighborSearchDataStructure(particleState, config):
    return buildDataStructure(particleState['positions'], particleState['supports'] * config['neighborhood']['verletScale'], particleState['supports'].max(), config['domain']['minExtent'], config['domain']['maxExtent'], config['domain']['periodic'], particleState['positions'].shape[0], verbose = False)



def getPeriodicPosition(x, config):
    periodic = config['domain']['periodicity']
    minDomain = config['domain']['minExtent']
    maxDomain = config['domain']['maxExtent']

    # dim = config['domain']['dim']
    # kernel = config['kernel']['function']

    with record_function("NeighborSearch [adjust Domain]"):
        periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
        if isinstance(periodic, torch.Tensor):
            periodicity = periodic
        if isinstance(periodic, bool):
            periodicity = torch.tensor([periodic] * x.shape[1], dtype = torch.bool).to(x.device)
        # if minDomain is not None and isinstance(minDomain, list):
            # minD = torch.tensor(minDomain).to(x.device).type(x.dtype)
        # else:
        minD = minDomain
        # if maxDomain is not None and isinstance(minDomain, list):
            # maxD = torch.tensor(maxDomain).to(x.device).type(x.dtype)
        # else:
        maxD = maxDomain
    with record_function("NeighborSearch [periodicity]"):
        return torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    

def computeNeighborhood(neighborhood, pos_x, pos_y, h_i, h_j, config, mode):
    i,j = neighborhood

    hij = None
    if mode == 'scatter':
        hij = h_j[j]
    elif mode == 'gather':
        hij = h_i[i]
    elif mode == 'symmetric':
        hij = 0.5 * (h_i[i] + h_j[j])


    xij = pos_x[i,:] - pos_y[j,:]
    xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], config['domain']['minExtent'][i], config['domain']['maxExtent'][i]) for i, periodic_i in enumerate(config['domain']['periodicity'])], dim = -1)
    # rij = torch.sqrt((xij**2).sum(-1))
    rij = torch.linalg.norm(xij, dim = -1)
    xij = xij / (rij + 1e-7).view(-1,1)

    rij = rij / hij

    mask = rij <= 1.0
    rij = rij[mask]
    xij = xij[mask,:]
    iFiltered = i[mask]
    jFiltered = j[mask]
    hijFiltered = hij[mask]

    return (iFiltered, jFiltered), hijFiltered, rij, xij

def updateDataStructure(referenceState, config, priorDatastructure, verbose = False):
    if priorDatastructure['referencePositions'].shape != referenceState['positions'].shape:
        if verbose:
            print(f'Updating Datastructure because of shape mismatch ({priorDatastructure["referencePositions"].shape} != {referenceState["positions"].shape})')
        return getNeighborSearchDataStructure(referenceState, config), True

    maxDistance = torch.linalg.norm(priorDatastructure['referencePositions'] - referenceState['positions'], dim = -1).max()
    minSupport = priorDatastructure['referenceSupports'].min()

    if maxDistance * 2 > (config['neighborhood']['verletScale'] - 1) * minSupport:    
        if verbose:
            print(f'Updating Datastructure because of distance mismatch ({maxDistance} > {(config["neighborhood"]["verletScale"] - 1) * minSupport})')
        return getNeighborSearchDataStructure(referenceState, config), True

    priorDatastructure['sortedPositions'] = referenceState['positions'][priorDatastructure['sortIndex'],:]
    priorDatastructure['sortedSupports'] = referenceState['supports'][priorDatastructure['sortIndex']]
    return priorDatastructure, False


from diffSPH.v2.compiler import compileSourceFiles


# computeNeighborhood_module = compileSourceFiles(
    # ['./cppSrc/computeNeighborhood.cpp', './cppSrc/computeNeighborhood.cu'], module_name = 'computeNeighborhood', verbose = False, openMP = True, verboseCuda = False, cuda_arch = None)
# from torch.utils.cpp_extension import load

# computeNeighborhood_cpp = computeNeighborhood_module.computeNeighborhood


def neighborSearch(queryState, referenceState, config, computeKernels = True, priorState = None, neighborDatastructure = None, verbose = False):
    if neighborDatastructure is None:
        if verbose:
            print(f'Building Datastructure because prior state is None')
        neighborDatastructure = getNeighborSearchDataStructure(referenceState, config)
        dirty = True
    else:
        neighborDatastructure, dirty = updateDataStructure(referenceState, config, neighborDatastructure, verbose)
    if neighborDatastructure is None:

        if verbose:
            print(f'Building Neighborlist because prior state is None')
        dirty = True
    
    if priorState is not None:
        if priorState['initialPositions'][0].shape != queryState['positions'].shape:
            if verbose:
                print(f'Updating Neighborsearch because of shape mismatch ({priorState["initialPositions"][0].shape} != {queryState["positions"].shape})')
            dirty = True
        else:

            maxDistance = torch.linalg.norm(priorState['initialPositions'][0] - queryState['positions'], dim = -1).max()
            minSupport = queryState['supports'].min()
            if maxDistance * 2 > (config['neighborhood']['verletScale'] - 1) * minSupport:    
                if verbose:
                    print(f'Updating Neighborsearch because of distance mismatch ({maxDistance} > {(config["neighborhood"]["verletScale"] - 1) * minSupport})')
                dirty = True
        
    else:
        if verbose:
            print(f'Updating Neighborsearch because priorState is None')
        dirty = True


    if dirty:
        neighborhood = neighborSearchExisting(queryState['positions'], queryState['supports'], neighborDatastructure, 'scatter', 1, 'cpp')

        numNeighbors_full = countUniqueEntries(neighborhood[0], queryState['positions'])[1].to(torch.int32)
        neighborOffsets_full = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors_full.device), torch.cumsum(numNeighbors_full, dim = 0).to(torch.int32)))[:-1]\

    
    else:
        neighborhood = priorState['fullIndices']
        numNeighbors_full = priorState['fullNumNeighbors']
        neighborOffsets_full = priorState['fullNeighborOffsets']

    pos_x = getPeriodicPosition(queryState['positions'], config)
    pos_y = getPeriodicPosition(referenceState['positions'], config)

    h_i = queryState['supports']
    h_j = referenceState['supports']

    neighborhood_actual, hij_actual, rij, xij = computeNeighborhood(neighborhood, pos_x, pos_y, h_i, h_j, config, 'scatter')

    numNeighbors = countUniqueEntries(neighborhood_actual[0], pos_x)[1].to(torch.int32)
    neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors.device), torch.cumsum(numNeighbors, dim = 0).to(torch.int32)))[:-1]\

    # numNeighbors, neighborOffsets, i, j, rij, xij, hij_actual = computeNeighborhood_cpp(
    # neighborhood, 
    # pos_x.shape[0],
    # numNeighbors_full, neighborOffsets_full,
    # (pos_x, pos_y), 
    # (h_i, h_j),
    # config['domain']['minExtent'], config['domain']['maxExtent'], config['domain']['periodicity'])

    # neighborhood_actual = (i, j)
# torch.cuda.synchronize()

    if dirty:
        neighborDict = {
            'indices': neighborhood_actual,
            'fullIndices': neighborhood,
            'distances': rij,
            'vectors': xij,
            'supports': hij_actual,
            'initialPositions': (queryState['positions'], referenceState['positions']),
            'numNeighbors': numNeighbors,
            'neighborOffsets': neighborOffsets,
            'fullNumNeighbors': numNeighbors_full,
            'fullNeighborOffsets': neighborOffsets_full
        }
    else:
        neighborDict = {
            'indices': neighborhood_actual,
            'fullIndices': priorState['fullIndices'],
            'distances': rij,
            'vectors': xij,
            'supports': hij_actual,
            'initialPositions': priorState['initialPositions'],
            'numNeighbors': numNeighbors,
            'neighborOffsets': neighborOffsets,
            'fullNumNeighbors': numNeighbors_full,
            'fullNeighborOffsets': neighborOffsets_full
        }


    if computeKernels:
        dim = config['domain']['dim']
        kernel = config['kernel']['function']
        neighborDict['kernels'] = kernel.kernel(rij, hij_actual, dim)
        neighborDict['gradients'] = kernel.kernelGradient(rij, xij, hij_actual, dim) 
    
    return neighborDatastructure, neighborDict

# ds, neighborhood = neighborSearch(perennialState['fluid'], perennialState['fluid'], config, True, None, None, True)



from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('neighborhood', 'scheme', str, 'compact', required = False, export = True),
    Parameter('neighborhood', 'verletScale', float, 1.5, required = False, export = True),
    ]
