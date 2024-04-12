import torch
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.math import scatter_sum
from torch.profiler import record_function


def computeAlpha(simulationState, config):
        fluidNeighbors = simulationState['fluid']['neighborhood']
        (i, j) = fluidNeighbors['indices']

        grad = fluidNeighbors['gradients']
        grad2 = torch.einsum('nd, nd -> n', grad, grad)

        term1 = simulationState['fluid']['actualArea'][j][:,None] * grad
        term2 = (simulationState['fluid']['actualArea']**2 / (simulationState['fluid']['areas'] * config['fluid']['rho0']))[j] * grad2

        kSum1 = scatter_sum(term1, i, dim=0, dim_size=simulationState['fluid']['areas'].shape[0])
        kSum2 = scatter_sum(term2, i, dim=0, dim_size=simulationState['fluid']['areas'].shape[0])

        fac = - config['timestep']['dt'] **2 * simulationState['fluid']['actualArea']
        mass = simulationState['fluid']['areas'] * config['fluid']['rho0']
        alpha = fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2
        # alpha = torch.clamp(alpha, -1, -1e-7)

        return alpha


def computeSourceTerm(simulationState, config):
    fac = config['timestep']['dt'] 
    div = sphOperationStates(simulationState['fluid'], simulationState['fluid'], (simulationState['fluid']['predictedVelocities'], simulationState['fluid']['predictedVelocities']), operation = 'divergence', gradientMode = 'difference', neighborhood = simulationState['fluid']['neighborhood'])
    return fac * div

def computePressureAcceleration(simulationState, config):
    stateA = simulationState['fluid']
    stateB = simulationState['fluid']
    neighborhood = stateA['neighborhood']

    return -sphOperationStates(stateA, stateB, (stateA['pressureB'], stateB['pressureB']), operation = 'gradient', gradientMode='summation', neighborhood= neighborhood) / stateA['densities'].view(-1,1)
    
def updatePressure(simulationState, config):
    stateA = simulationState['fluid']
    stateB = simulationState['fluid']
    neighborhood = stateA['neighborhood']

    dt = config['timestep']['dt']
    kernelSum = -dt**2 * sphOperationStates(stateA, stateB, (stateA['pressureAccel'], stateB['pressureAccel']), operation = 'divergence', gradientMode='difference', neighborhood= neighborhood)

    sourceTerm = stateA['sourceTerm']
    residual = kernelSum - sourceTerm
    pressure = stateA['pressureA'] + 0.3 * (sourceTerm - kernelSum) / stateA['alpha']
    pressure = torch.max(pressure, torch.zeros_like(pressure))

    return pressure, residual

def dfsphSolve(simulationState, config):
    advection_acceleration = simulationState['fluid']['gravityAccel'] + simulationState['fluid']['velocityDiffusion']

    simulationState['fluid']['predictedAcceleration'] = torch.zeros_like(simulationState['fluid']['velocities'])    
    simulationState['fluid']['predictedVelocities'] = simulationState['fluid']['velocities'] + config['timestep']['dt'] * advection_acceleration

    simulationState['fluid']['actualArea'] = simulationState['fluid']['masses'] / simulationState['fluid']['densities']

    simulationState['fluid']['alpha'] = computeAlpha(simulationState, config)

    simulationState['fluid']['sourceTerm'] = 1 - simulationState['fluid']['areas']  / simulationState['fluid']['actualArea'] + computeSourceTerm(simulationState, config)

    simulationState['fluid']['pressureB'] = torch.zeros(simulationState['fluid']['numParticles'], device = config['compute']['device'])
    simulationState['fluid']['pressureA'] = torch.zeros(simulationState['fluid']['numParticles'], device = config['compute']['device'])


    errors = []
    pressures = []
    i = 0
    error = 0.
    minIters = 2
    maxIters = 256
    errorThreshold = 1e-3


    while i < maxIters and (i < minIters or error > errorThreshold):
        simulationState['fluid']['pressureAccel'] = computePressureAcceleration(simulationState, config)
        simulationState['fluid']['pressureA'][:] = simulationState['fluid']['pressureB'].clone()
        
        simulationState['fluid']['pressureB'], simulationState['fluid']['residual'] = updatePressure(simulationState, config)
        error = torch.mean(torch.clamp(simulationState['fluid']['residual'], min = -errorThreshold))
        
        errors.append(error.detach().cpu().item())
        # print(f'{i:2d} -> {error.detach().cpu().item():+.4e}, pressure mean: {simulationState["fluid"]["pressureB"].mean().detach().cpu().item():+.4e}')
        # break
        pressures.append(simulationState['fluid']['pressureB'].mean().detach().cpu().item())

        i += 1

    return errors, pressures, computePressureAcceleration(simulationState, config), simulationState['fluid']['pressureB']