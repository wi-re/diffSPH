import torch
from diffSPH.v2.sphOps import sphOperationStates, sphOperation
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from torch.profiler import record_function


def renormalizedDensityGradient(stateA, stateB, neighborhood, simConfig):    
    with record_function("[SPH] - Density Gradient Renormalization (nabla^L rho_i)"):
        (i, j) = neighborhood['indices']

        gradKernel = neighborhood['gradients']
        Ls = stateA['fluidL'][i]

        normalizedGradients = torch.einsum('ijk,ik->ij', Ls, gradKernel)

        return sphOperation(
            (stateA['fluidMasses'], stateB['fluidMasses']), 
            (stateA['fluidDensities'], stateB['fluidDensities']),
            (stateA['fluidDensities'], stateB['fluidDensities']),
            neighborhood['indices'], 
            neighborhood['kernels'], normalizedGradients,
            neighborhood['distances'], neighborhood['vectors'], neighborhood['supports'], 
            stateA['numParticles'], 
            operation = 'gradient', gradientMode = 'difference', divergenceMode = 'div', 
            kernelLaplacians = neighborhood['laplacians'] if 'laplacians' in neighborhood else None)


def computeDensityDeltaTerm(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Density Diffusion (delta-SPH)"):
        (i, j) = neighborhood['indices']
        scheme = simConfig['diffusion']['densityScheme']

        rij = neighborhood['distances'] * neighborhood['supports']
        if scheme == 'deltaSPH':
            grad_ij = stateA['fluidGradRho^L'][i] + stateB['fluidGradRho^L'][j]
            rho_ij = 2 * (stateB['fluidDensities'][j] - stateA['fluidDensities'][i]) / (rij + 1e-6 * neighborhood['supports'])
            psi_ij = -rho_ij.view(-1,1) * neighborhood['vectors'] - grad_ij
        elif scheme == 'denormalized':
            grad_ij = stateA['fluidGradRho'][i] + stateB['fluidGradRho'][j]
            rho_ij = 2 * (stateB['fluidDensities'][j] - stateA['fluidDensities'][i]) / (rij + 1e-6 * neighborhood['supports'])
            psi_ij = -rho_ij.view(-1,1) * neighborhood['vectors'] - grad_ij
        elif scheme == 'densityOnly':
            rho_ij = 2 * (stateB['fluidDensities'][j] - stateA['fluidDensities'][i]) / (rij + 1e-6 * neighborhood['supports'])
            psi_ij = -rho_ij.view(-1,1) * neighborhood['vectors']

        return simConfig['diffusion']['delta'] * stateA['fluidSupports'] / simConfig['kernel']['kernelScale'] * simConfig['fluid']['cs'] * sphOperationStates(stateA, stateB, psi_ij, operation = 'divergence', gradientMode='difference')




    
from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('diffusion', 'delta', float, 0.1, required = False,export = False, hint = 'Delta value for the density diffusion term'),
        Parameter('diffusion', 'densityScheme', str, 'deltaSPH', required = False,export = False, hint = 'Scheme for the density diffusion term') 
    ]