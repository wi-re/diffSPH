from diffSPH.v2.plotting import setPlotBaseAttributes
import torch
import numpy as np
from diffSPH.v2.modules.neighborhood import neighborSearch

def getModPositions(x, config):
    periodicity = config['domain']['periodicity']
    minDomain = config['domain']['minExtent']
    maxDomain = config['domain']['maxExtent']
    # periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
    mod_positions = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    return mod_positions

def processOutlet(region, config, perennialState):
    mod_positions = perennialState['fluid']['positions']

    outletSDF = region['sdf']
    dist = outletSDF(mod_positions)
    mask = dist < 0
    reducedIndices = torch.arange(perennialState['fluid']['positions'].shape[0], device = perennialState['fluid']['positions'].device, dtype =  torch.int64)[~mask]

    if mask.sum() == 0:
        return
    # print(f'Outlet: {mask.sum()} particles removed')
    # print('Current particles:', perennialState['fluid']['positions'].shape[0])
    # print(reducedIndices.shape, reducedIndices)

    for k in perennialState['fluid'].keys():
        if isinstance(perennialState['fluid'][k], torch.Tensor):
            perennialState['fluid'][k] = perennialState['fluid'][k][reducedIndices]
    
    perennialState['fluid']['numParticles'] = perennialState['fluid']['positions'].shape[0]

def processForcing(region, config, perennialState):
    mod_positions = perennialState['fluid']['positions']
    outletSDF = region['sdf']
    dist = outletSDF(mod_positions)
    mask = dist < 0
    reducedIndices = torch.arange(perennialState['fluid']['positions'].shape[0], device = perennialState['fluid']['positions'].device, dtype =  torch.int64)[mask]

    if mask.sum() == 0:
        return
    
    perennialState['fluid']['velocities'][reducedIndices,0] = region['velocity'][0]
    perennialState['fluid']['velocities'][reducedIndices,1] = region['velocity'][1]
    perennialState['fluid']['densities'][reducedIndices] = config['fluid']['rho0']



def mergeStates(oldState, newState):
    for key in oldState.keys():
        if not isinstance(oldState[key], torch.Tensor):
            continue
        if key in newState.keys():
            # print(f'Merging key {key} with shape {oldState[key].shape} and {newState[key].shape}')
            oldState[key] = torch.cat([oldState[key], newState[key]], dim = 0)
        else:
            # print(f'Key {key} not found in newState')

            pseudoVar = torch.zeros([newState['numParticles'], *oldState[key].shape[1:]], dtype = oldState[key].dtype, device = oldState[key].device)
            # print(pseudoVar.shape)

            oldState[key] = torch.cat([oldState[key], pseudoVar], dim = 0)
    oldState['numParticles'] = oldState['positions'].shape[0]#.detach().cpu().item()
    return oldState

from diffSPH.v2.finiteDifference import continuousGradient, centralDifferenceStencil
from diffSPH.v2.modules.inletOutlet import continuousGradient, centralDifferenceStencil
def buildOutletGhostParticles(regions, perennialState, config):
    ghostState = None

    for region in regions:
        if region['type'] != 'mirror':
            continue
            # ghostState = copy.deepcopy(region)
        outletSDF = region['sdf']
        mod_positions = perennialState['fluid']['positions']
        dist = outletSDF(mod_positions)
        mask = dist < 0
        reducedIndices = torch.arange(perennialState['fluid']['positions'].shape[0], device = perennialState['fluid']['positions'].device, dtype =  torch.int64)[mask]

        if mask.sum() == 0: 
            continue

    # axis[0,0].scatter(perennialState['fluid']['positions'][:,0].detach().cpu().numpy(), perennialState['fluid']['positions'][:,1].detach().cpu().numpy(), s = 1, c = mask)

        i = reducedIndices
        j = torch.arange(i.shape[0], device = i.device, dtype = i.dtype)
        pos = perennialState['fluid']['positions'][reducedIndices]

        mod_pos = getModPositions(pos, config)

        dist = outletSDF(mod_pos)

        grad = continuousGradient(outletSDF, pos, stencil = centralDifferenceStencil(1,2), dx = config['particle']['support']*0.01, order = 1)
        grad = grad / (torch.linalg.norm(grad, dim = 1, keepdim = True) + 1e-7)
        ghostPos = pos - 2 * dist[:,None] * grad

        if ghostState is None:
            ghostState = {
                'positions': ghostPos,
                'velocities': torch.zeros_like(ghostPos),
                'supports': torch.ones(ghostPos.shape[0], device = ghostPos.device) * config['particle']['support'],
                'densities' : torch.ones(ghostPos.shape[0], device = ghostPos.device) * config['fluid']['rho0'],
                'masses': torch.ones(ghostPos.shape[0], device = ghostPos.device) * config['particle']['volume'] * config['fluid']['rho0'],
                'index': j,
                'fluidIndex': i,
                'numParticles': j.shape[0],
                'sdf_dist': dist,
                'sdf_grad': grad
            }
        else:
            ghostState['positions'] = torch.cat([ghostState['positions'], ghostPos], dim = 0)
            ghostState['velocities'] = torch.cat([ghostState['velocities'], torch.zeros_like(ghostPos)], dim = 0)
            ghostState['supports'] = torch.cat([ghostState['supports'], torch.ones(ghostPos.shape[0], device = ghostPos.device) * config['particle']['support']], dim = 0)
            ghostState['densities'] = torch.cat([ghostState['densities'], torch.ones(ghostPos.shape[0], device = ghostPos.device) * config['fluid']['rho0']], dim = 0)
            ghostState['index'] = torch.cat([ghostState['index'], j], dim = 0)
            ghostState['fluidIndex'] = torch.cat([ghostState['fluidIndex'], i], dim = 0)
            ghostState['numParticles'] = ghostState['positions'].shape[0]
            ghostState['sdf_dist'] = torch.cat([ghostState['sdf_dist'], dist], dim = 0)
            ghostState['sdf_grad'] = torch.cat([ghostState['sdf_grad'], grad], dim = 0)

    if ghostState is None:
        return None

    gridConfig = {
        'domain': config['domain'],
        'simulation': {
            'supportScheme': 'scatter'
        },
        'neighborhood':{
            'algorithm': 'compact',
            'scheme': 'compact',
            'verletScale': 1.0
        },
        'kernel': config['kernel']
    }
    # print('...')
    # ghostState['neighborhood'] = neighborSearch(ghostState, perennialState['fluid'], gridConfig, computeKernels=False)
    _, ghostState['neighborhood'] = neighborSearch(ghostState, perennialState['fluid'], gridConfig, computeKernels=True)
    # _, ghostState['numNeighbors'] = countUniqueEntries(ghostState['neighborhood']['indices'][0], ghostState['positions'])
    ghostState['numNeighbors'] = ghostState['neighborhood']['numNeighbors']

    return ghostState


from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.sphOps import adjunctMatrix, LiuLiuConsistent

def processInlet(perennialState, config, emitter):
    emitterState = emitter['particles']

    _, emitterNeighborhood = neighborSearch(emitterState, perennialState['fluid'], config, computeKernels=False)

    distance = emitterNeighborhood['distances']
    i = emitterNeighborhood['indices'][0]
    newDistance = distance.new_ones(emitterState['numParticles'], dtype = config['compute']['dtype']) #* config['particle']['support']
    minDistance = newDistance.index_reduce_(dim = 0, index = i, source = distance, include_self = False, reduce = 'amin')

    emitterMask = minDistance >= config['particle']['dx'] / config['particle']['support']

    newPositions = emitterState['positions'][emitterMask].to(config['compute']['device'])
    newParticleState = {
        'numParticles': newPositions.shape[0],#.detach().cpu(),
        'positions': newPositions,

        'areas': torch.ones(newPositions.shape[0], dtype = config['compute']['dtype'], device = config['compute']['device']) * config['particle']['volume'],
        'masses': torch.ones(newPositions.shape[0], dtype = config['compute']['dtype'], device = config['compute']['device']) * config['particle']['volume'] * config['fluid']['rho0'],
        'densities': torch.ones(newPositions.shape[0], dtype = config['compute']['dtype'], device = config['compute']['device']) * config['fluid']['rho0'],
        'supports': torch.ones(newPositions.shape[0], dtype = config['compute']['dtype'], device = config['compute']['device']) * config['particle']['support'],

        'pressures': torch.zeros(newPositions.shape[0], dtype = config['compute']['dtype'], device = config['compute']['device']),
        'divergence': torch.zeros(newPositions.shape[0], dtype = config['compute']['dtype'], device = config['compute']['device']),

        'velocities': torch.zeros(newPositions.shape, dtype = config['compute']['dtype'], device = config['compute']['device']),
        'accelerations': torch.zeros(newPositions.shape, dtype = config['compute']['dtype'], device = config['compute']['device']),
        'index': torch.arange(newPositions.shape[0], dtype = torch.int32, device = config['compute']['device']) + perennialState['uidCounter'],
        # 'neighbors': None,
    }
    newParticleState['velocities'][:,0] = emitter['velocity'][0]
    newParticleState['velocities'][:,1] = emitter['velocity'][1]
    
    if 'initialPositions' in perennialState['fluid']:
        # print('...', newParticleState['positions'])
        newParticleState['initialPositions'] = newParticleState['positions'].clone()

    # print(f'Adding {newParticleState["numParticles"]} particles (total {perennialState["fluid"]["numParticles"] + newParticleState["numParticles"]})')
    perennialState['uidCounter'] += newParticleState['numParticles']

    perennialState['fluid'] = mergeStates(perennialState['fluid'], newParticleState)