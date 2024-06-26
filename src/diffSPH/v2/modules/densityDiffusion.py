import torch
from diffSPH.v2.sphOps import sphOperationStates, sphOperation
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from torch.profiler import record_function


def renormalizedDensityGradient(stateA, stateB, neighborhood, simConfig):    
    with record_function("[SPH] - Density Gradient Renormalization (nabla^L rho_i)"):
        (i, j) = neighborhood['indices']

        gradKernel = neighborhood['gradients']
        Ls = stateA['L'][i]

        normalizedGradients = torch.einsum('ijk,ik->ij', Ls, gradKernel)
        # normalizedGradients = stateA['normGrad']

        return sphOperation(
            (stateA['masses'], stateB['masses']), 
            (stateA['densities'], stateB['densities']),
            (stateA['densities'], stateB['densities']),
            neighborhood['indices'], 
            neighborhood['kernels'], normalizedGradients,
            neighborhood['distances'], neighborhood['vectors'], neighborhood['supports'], 
            stateA['numParticles'], 
            operation = 'gradient', gradientMode = 'difference', divergenceMode = 'div', 
            kernelLaplacians = neighborhood['laplacians'] if 'laplacians' in neighborhood else None)

def densityGradient(stateA, stateB, neighborhood, simConfig):    
    with record_function("[SPH] - Density Gradient Renormalization (nabla^L rho_i)"):
        (i, j) = neighborhood['indices']

        gradKernel = neighborhood['gradients']
        # Ls = stateA['L'][i]

        # normalizedGradients = torch.einsum('ijk,ik->ij', Ls, gradKernel)

        return sphOperation(
            (stateA['masses'], stateB['masses']), 
            (stateA['densities'], stateB['densities']),
            (stateA['densities'], stateB['densities']),
            neighborhood['indices'], 
            neighborhood['kernels'], gradKernel,
            neighborhood['distances'], neighborhood['vectors'], neighborhood['supports'], 
            stateA['numParticles'], 
            operation = 'gradient', gradientMode = 'difference', divergenceMode = 'div', 
            kernelLaplacians = neighborhood['laplacians'] if 'laplacians' in neighborhood else None)


def computeDensityDeltaTerm(stateA, stateB, neighborhood, simConfig, schemeOverride = None):
    with record_function("[SPH] - Density Diffusion (delta-SPH)"):
        (i, j) = neighborhood['indices']
        scheme = simConfig['diffusion']['densityScheme'] if schemeOverride is None else schemeOverride

        rij = neighborhood['distances'] * neighborhood['supports']
        if scheme == 'deltaSPH':
            grad_ij = stateA['gradRho^L'][i] + stateB['gradRho^L'][j]
            rho_ij = 2 * (stateB['densities'][j] - stateA['densities'][i]) / (rij + 1e-6 * neighborhood['supports'])
            psi_ij = -rho_ij.view(-1,1) * neighborhood['vectors'] + grad_ij
        elif scheme == 'denormalized':
            grad_ij = stateA['gradRho'][i] + stateB['gradRho'][j]
            rho_ij = 2 * (stateB['densities'][j] - stateA['densities'][i]) / (rij + 1e-6 * neighborhood['supports'])
            psi_ij = -rho_ij.view(-1,1) * neighborhood['vectors'] - grad_ij
        elif scheme == 'densityOnly':
            rho_ij = 2 * (stateB['densities'][j] - stateA['densities'][i]) / (rij + 1e-6 * neighborhood['supports'])
            psi_ij = -rho_ij.view(-1,1) * neighborhood['vectors']
        elif scheme == 'deltaOnly':
            grad_ij = stateA['gradRho^L'][i] + stateB['gradRho^L'][j]
            psi_ij = -grad_ij
        elif scheme == 'denormalizedOnly':
            grad_ij = stateA['gradRho'][i] + stateB['gradRho'][j]
            psi_ij = -grad_ij

        return simConfig['diffusion']['delta'] * stateA['supports'] / simConfig['kernel']['kernelScale'] * simConfig['fluid']['cs'] * sphOperationStates(stateA, stateB, psi_ij, operation = 'divergence', gradientMode='difference', neighborhood = neighborhood)




    
from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('diffusion', 'delta', float, 0.1, required = False,export = False, hint = 'Delta value for the density diffusion term'),
        Parameter('diffusion', 'densityScheme', str, 'deltaSPH', required = False,export = False, hint = 'Scheme for the density diffusion term') 
    ]