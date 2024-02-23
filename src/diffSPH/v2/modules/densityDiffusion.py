import torch
from diffSPH.v2.sphOps import sphOperationFluidState, sphOperation
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices


def renormalizedDensityGradient(simulationState, config):
    (i, j) = simulationState['fluidNeighborhood']['indices']

    gradKernel = simulationState['fluidNeighborhood']['gradients']
    Ls = simulationState['fluidL'][i]

    normalizedGradients = torch.einsum('ijk,ik->ij', Ls, gradKernel)

    return sphOperation(
        (simulationState['fluidMasses'], simulationState['fluidMasses']), 
        (simulationState['fluidDensities'], simulationState['fluidDensities']),
        (simulationState['fluidDensities'], simulationState['fluidDensities']),
        simulationState['fluidNeighborhood']['indices'], 
        simulationState['fluidNeighborhood']['kernels'], normalizedGradients,
        simulationState['fluidNeighborhood']['distances'], simulationState['fluidNeighborhood']['vectors'], simulationState['fluidNeighborhood']['supports'], 
        simulationState['numParticles'], 
        operation = 'gradient', gradientMode = 'difference', divergenceMode = 'div', 
        kernelLaplacians = simulationState['fluidNeighborhood']['laplacians'] if 'laplacians' in simulationState['fluidNeighborhood'] else None)


def computeDensityDeltaTerm(fluidState, config):
    (i, j) = fluidState['fluidNeighborhood']['indices']
    scheme = config['diffusion']['densityScheme']

    rij = fluidState['fluidNeighborhood']['distances'] * fluidState['fluidNeighborhood']['supports']
    if scheme == 'deltaSPH':
        grad_ij = fluidState['fluidGradRho^L'][i] + fluidState['fluidGradRho^L'][j]
        rho_ij = 2 * (fluidState['fluidDensities'][j] - fluidState['fluidDensities'][i]) / (rij + 1e-6 * fluidState['fluidSupports'])
        psi_ij = -rho_ij.view(-1,1) * fluidState['fluidNeighborhood']['vectors'] - grad_ij
    elif scheme == 'denormalized':
        grad_ij = fluidState['fluidGradRho'][i] + fluidState['fluidGradRho'][j]
        rho_ij = 2 * (fluidState['fluidDensities'][j] - fluidState['fluidDensities'][i]) / (rij + 1e-6 * fluidState['fluidSupports'])
        psi_ij = -rho_ij.view(-1,1) * fluidState['fluidNeighborhood']['vectors'] - grad_ij
    elif scheme == 'densityOnly':
        rho_ij = 2 * (fluidState['fluidDensities'][j] - fluidState['fluidDensities'][i]) / (rij + 1e-6 * fluidState['fluidSupports'])
        psi_ij = -rho_ij.view(-1,1) * fluidState['fluidNeighborhood']['vectors']

    return config['diffusion']['delta'] * fluidState['fluidSupports'] * config['fluid']['cs'] * sphOperationFluidState(fluidState, psi_ij, operation = 'divergence', gradientMode='difference')




    
from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('diffusion', 'delta', float, 0.1, required = False,export = False, hint = 'Delta value for the density diffusion term'),
        Parameter('diffusion', 'densityScheme', str, 'deltaSPH', required = False,export = False, hint = 'Scheme for the density diffusion term') 
    ]