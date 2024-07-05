import torch
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.math import scatter_sum
from torch.profiler import record_function


def computeAlpha(stateA, stateB, config, neighborhood, density = True):
    dt = config['timestep']['dt']**2 if density else config['timestep']['dt']

    fluidNeighbors = neighborhood# simulationState['fluid']['neighborhood']
    (i, j) = fluidNeighbors['indices']

    grad = fluidNeighbors['gradients']
    grad2 = torch.einsum('nd, nd -> n', grad, grad)

    term1 = stateB['actualArea'][j][:,None] * grad
    term2 = (stateB['actualArea']**2 / (stateB['areas'] * config['fluid']['rho0']))[j] * grad2

    kSum1 = scatter_sum(term1, i, dim=0, dim_size=stateA['areas'].shape[0])
    kSum2 = scatter_sum(term2, i, dim=0, dim_size=stateA['areas'].shape[0])

    fac = - dt * stateA['actualArea']
    mass = stateA['areas'] * config['fluid']['rho0']
    alpha = fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2
    # alpha = torch.clamp(alpha, -1, -1e-7)

    return alpha


def computeSourceTerm(stateA, stateB, config, neighborhood, density = True):
    fac = config['timestep']['dt'] if density else 1.0
    div = sphOperationStates(stateA, stateB, (stateA['predictedVelocities'], stateB['predictedVelocities']), operation = 'divergence', gradientMode = 'difference', neighborhood = neighborhood)
    return fac * div

def computePressureAcceleration(stateA, stateB, config, neighborhood):

    return -sphOperationStates(stateA, stateB, (stateA['pressureB'], stateB['pressureB']), operation = 'gradient', gradientMode='summation', neighborhood= neighborhood) / stateA['densities'].view(-1,1)
    
def updatePressure(stateA, stateB, config, neighborhood, density = True):

    dt = config['timestep']['dt']**2 if density else config['timestep']['dt']
    kernelSum = -dt * sphOperationStates(stateA, stateB, (stateA['pressureAccel'], stateB['pressureAccel']), operation = 'divergence', gradientMode='difference', neighborhood= neighborhood)

    sourceTerm = stateA['sourceTerm']
    residual = kernelSum - sourceTerm
    pressure = stateA['pressureA'] + config['dfsph']['omega'] * (sourceTerm - kernelSum) / stateA['alpha']
    if config['dfsph']['clampPressure']:
        pressure = torch.max(pressure, torch.zeros_like(pressure))

    return pressure, residual

def dfsphSolve(stateA, stateB, neighborhood, config):
    solveDensity = config['dfsph']['sourceTerm'] == 'density'
    advection_acceleration = stateA['advection']#stateA['gravityAccel'] + stateA['velocityDiffusion']

    stateA['predictedAcceleration'] = torch.zeros_like(stateA['velocities'])    
    stateA['predictedVelocities'] = stateA['velocities'] + config['timestep']['dt'] * advection_acceleration

    stateA['actualArea'] = stateA['masses'] / stateA['densities']


    if config['dfsph']['sourceTerm'] == 'density':
        stateA['sourceTerm'] = 1 - stateA['areas']  / stateA['actualArea'] - computeSourceTerm(stateA, stateB, config, neighborhood, density = solveDensity)
    else:
        stateA['sourceTerm'] = computeSourceTerm(stateA, stateB, config, neighborhood, density = solveDensity) #/ config['timestep']['dt']

    stateA['pressureB'] = torch.zeros(stateA['numParticles'], device = config['compute']['device'])
    stateA['pressureA'] = torch.zeros(stateA['numParticles'], device = config['compute']['device'])

    if 'pressureIncompressible' in stateA and solveDensity:
        stateA['pressureB'] = 0.75 * stateA['pressureIncompressible']
        stateA['pressureA'] = 0.75 * stateA['pressureIncompressible']
    elif 'pressureDivergence' in stateA and not solveDensity:
        stateA['pressureB'] = 0.75 * stateA['pressureDivergence']
        stateA['pressureA'] = 0.75 * stateA['pressureDivergence']

    errors = []
    pressures = []
    i = 0
    error = 0.
    minIters = config['dfsph']['minIters']
    maxIters = config['dfsph']['maxIters']
    errorThreshold = config['dfsph']['errorThreshold']
   # fac = config['timestep']['dt'] if config['dfsph']['sourceTerm'] == 'divergence' else 1.0

    stateA['alpha'] = computeAlpha(stateA, stateB, config, neighborhood, density = solveDensity)# / fac

    while i < maxIters and (i < minIters or error > errorThreshold):
        stateA['pressureAccel'] = computePressureAcceleration(stateA, stateB, config, neighborhood)
        stateA['pressureA'][:] = stateA['pressureB'].clone()
        
        stateA['pressureB'], stateA['residual'] = updatePressure(stateA, stateB, config, neighborhood, density = solveDensity)
        # if config['dfsph']['sourceTerm'] == 'density':
            # stateA['pressureB'] = torch.max(stateA['pressureB'], torch.zeros_like(stateA['pressureB']))
            # error = torch.mean(torch.clamp(stateA['residual'], min = -errorThreshold))
        error = torch.mean(torch.clamp(stateA['residual'] / config['fluid']['rho0'], min = -errorThreshold))
        
        errors.append(error.detach().cpu().item())
        # print(f'{i:2d} -> {error.detach().cpu().item():+.4e}, pressure mean: {stateA["pressureB"].mean().detach().cpu().item():+.4e}')
        # break
        pressures.append(stateA['pressureB'].mean().detach().cpu().item())

        i += 1

    a_p = computePressureAcceleration(stateA, stateB, config, neighborhood)

    stateA[f'convergence_{config["dfsph"]["sourceTerm"]}'] = errors

    return a_p


from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('dfsph', 'minIters', int, 2, required = False,export = False, hint = 'Minimum number of iterations'),
        Parameter('dfsph', 'maxIters', int, 256, required = False,export = False, hint = 'Maximum number of iterations'),
        Parameter('dfsph', 'errorThreshold', float, 1e-4, required = False,export = False, hint = 'Error threshold for pressure solver'),
        Parameter('dfsph', 'omega', float, 0.5, required = False,export = False, hint = 'Relaxation factor for pressure solver'),
        Parameter('dfsph', 'sourceTerm', str, 'density', required = False,export = False, hint = 'Source term for pressure solver'),
        Parameter('dfsph', 'clampPressure', bool, True, required = False,export = False, hint = 'Clamp pressure to positive values'),

    ]