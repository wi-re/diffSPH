import torch
from diffSPH.v2.sphOps import sphOperationStates
import numpy as np
from diffSPH.v2.modules.neighborhood import neighborSearch
from diffSPH.v2.sphOps import adjunctMatrix, LiuLiuConsistent


def buildBoundaryGhostParticles(perennialState, config):
    boundaryParticlePositions = perennialState['boundary']['positions'] 
    ghostParticlePositions = boundaryParticlePositions - 2 * perennialState['boundary']['distances'].view(-1,1) * perennialState['boundary']['normals'] 

    boundaryGhostState = {
        'positions': ghostParticlePositions,
        'areas': perennialState['boundary']['areas'],
        'masses': perennialState['boundary']['masses'],
        'densities': perennialState['boundary']['densities'],
        'supports': perennialState['boundary']['supports'],
        'velocities': perennialState['boundary']['velocities'],
        'numParticles': ghostParticlePositions.shape[0],
    }

    _, boundaryGhostState['neighborhood'] = neighborSearch(boundaryGhostState, perennialState['fluid'], config, True, perennialState['boundaryGhost']['neighborhood'] if 'boundaryGhost' in perennialState and 'neighborhood' in perennialState['boundaryGhost'] else None, perennialState['fluid']['datastructure'] if 'datastructure' in perennialState['fluid'] else None, False)
    boundaryGhostState['numNeighbors'] = boundaryGhostState['neighborhood']['numNeighbors']
    # boundaryGhostState['neighborhood'] = neighborSearch(boundaryGhostState, perennialState['fluid'], config, computeKernels=True)
    # _, boundaryGhostState['numNeighbors'] = countUniqueEntries(boundaryGhostState['neighborhood']['indices'][0], ghostParticlePositions)

    return boundaryGhostState


def mDBCDensity(perennialState, config):
    boundaryGhostState = perennialState['boundaryGhost']

    shepardNominator = sphOperationStates(boundaryGhostState, perennialState['fluid'], None, operation = 'density', neighborhood = boundaryGhostState['neighborhood'])
    shepardDenominator = sphOperationStates(boundaryGhostState, perennialState['fluid'], 
            (torch.ones_like(boundaryGhostState['masses']), torch.ones_like(perennialState['fluid']['masses'])),
            # (boundaryGhostState['densities'] / boundaryGhostState['masses'], perennialState['fluid']['densities'] / perennialState['fluid']['masses']),
              operation = 'interpolate', neighborhood = boundaryGhostState['neighborhood'])

    shepardDensity = shepardNominator / (shepardDenominator + 1e-7)

    gradientSum = sphOperationStates(boundaryGhostState, perennialState['fluid'], (boundaryGhostState['densities'], perennialState['fluid']['densities']), operation = 'gradient', neighborhood = boundaryGhostState['neighborhood'], gradientMode = 'naive')

    b = torch.hstack((shepardNominator[:,None], gradientSum))

    volumeSum = sphOperationStates(boundaryGhostState, perennialState['fluid'], (torch.ones_like(boundaryGhostState['masses']), torch.ones_like(perennialState['fluid']['masses'])), operation = 'interpolate', neighborhood = boundaryGhostState['neighborhood'])
    volumeGradSum = sphOperationStates(boundaryGhostState, perennialState['fluid'], (torch.ones_like(boundaryGhostState['masses']), torch.ones_like(perennialState['fluid']['masses'])), operation = 'gradient', neighborhood = boundaryGhostState['neighborhood'], gradientMode = 'naive')

    xij = -boundaryGhostState['neighborhood']['vectors'] * boundaryGhostState['neighborhood']['distances'].view(-1,1) * config['particle']['support']

    positionSum = sphOperationStates(boundaryGhostState, perennialState['fluid'], xij, operation = 'interpolate', neighborhood = boundaryGhostState['neighborhood'])
    positionMatrix = sphOperationStates(boundaryGhostState, perennialState['fluid'], xij, operation = 'gradient', neighborhood = boundaryGhostState['neighborhood'], gradientMode = 'naive')

    A_g = torch.zeros((boundaryGhostState['numParticles'], 3, 3), dtype = xij.dtype, device = xij.device)

    A_g[:,0,0] = volumeSum
    A_g[:,1,0] = volumeGradSum[:,0]
    A_g[:,2,0] = volumeGradSum[:,1]

    A_g[:,0,1] = positionSum[:,0]
    A_g[:,0,2] = positionSum[:,1]

    A_g[:,1,1] = positionMatrix[:,0,0]
    A_g[:,1,2] = positionMatrix[:,0,1]
    A_g[:,2,1] = positionMatrix[:,1,0]
    A_g[:,2,2] = positionMatrix[:,1,1]


    neighCounts = boundaryGhostState['numNeighbors']
    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]
    numPtcls = boundaryGhostState['numParticles']
    restDensity = config['fluid']['rho0']

    boundaryDensity = torch.ones(numPtcls, dtype = xij.dtype, device = xij.device) * restDensity
    boundaryDensity[neighCounts > 0] = shepardDensity[neighCounts > 0] #/ restDensity
    threshold = 5
    boundaryParticlePositions = perennialState['boundary']['positions']
    ghostParticlePositions = boundaryGhostState['positions']
    relPos = boundaryParticlePositions - ghostParticlePositions
    # relDist = torch.linalg.norm(relPos, dim = 1)
    # relDist = torch.clamp(relDist, min = 1e-7, max = config['particle']['support']*3.)
    # relPos = relPos * (relDist / (torch.linalg.norm(relPos, dim = 1) + 1e-7))[:,None]

    boundaryDensity[neighCounts > threshold] = (res[neighCounts > threshold,0] + torch.einsum('nu, nu -> n',(relPos)[neighCounts > threshold, :], res[neighCounts > threshold, 1:] ))
    # boundaryDensity = torch.clamp(boundaryDensity, min = restDensity)
    # self.fluidVolume = self.boundaryVolume / self.boundaryDensity

    solution, M, b = LiuLiuConsistent(boundaryGhostState, perennialState['fluid'], perennialState['fluid']['densities'])
    # boundaryDensity = 

    
    # boundaryDensity = torch.ones(numPtcls, dtype = xij.dtype, device = xij.device) * restDensity
    # boundaryDensity[neighCounts > 0] = shepardDensity[neighCounts > 0] #/ restDensity
    # threshold = 5

    relPos = boundaryParticlePositions - ghostParticlePositions
    # extrapolated = solution[:,0] + torch.einsum('nd, nd -> n', -relPos, solution[:,1:])

    # # neighCounts = boundaryGhostState['numNeighbors']
    # # boundaryDensity = shepardDensity
    # boundaryDensity[neighCounts > threshold] = torch.clamp(extrapolated[neighCounts > threshold], min = restDensity)
    boundaryDensity = torch.clamp(boundaryDensity, min = restDensity)

    # print(f'Boundary Density for Timestep {perennialState["timestep"]}: {boundaryDensity.min().item()} - {boundaryDensity.max().item()} mean: {boundaryDensity.mean().item()}')
    # print(f'shepardDensity: {shepardDensity.min().item()} - {shepardDensity.max().item()} mean: {shepardDensity.mean().item()}')
    # print(f'Shephard Nom: {shepardNominator.min().item()} - {shepardNominator.max().item()} mean: {shepardNominator.mean().item()}')
    # print(f'Shephard Denom: {shepardDenominator.min().item()} - {shepardDenominator.max().item()} mean: {shepardDenominator.mean().item()}')
    # print(f'neighCounts: {neighCounts.min().item()} - {neighCounts.max().item()} mean: {neighCounts.median().item()}')

    return boundaryDensity, shepardDensity