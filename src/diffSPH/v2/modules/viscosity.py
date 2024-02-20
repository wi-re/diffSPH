
import torch
import torch
from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum

from torch.profiler import record_function

def computeViscosityMonaghan(simulationState, config):
    with record_function("Viscosity [Monaghan]"):       
        return config['fluid']['mu'] * sphOperationFluidState(simulationState, (simulationState['fluidVelocities'], simulationState['fluidVelocities']), operation = 'laplacian', gradientMode='conserving')# / simulationState['fluidDensities'].view(-1,1)

# simulationState['viscosity'] = physicalParameters['nu'] * sphOperation(
#     (areas, areas), 
#     (simulationState['rho'], simulationState['rho']),
#     (u, u),
#     (i, j), kernel, gradKernel, rij, xij, hij, x.shape[0], operation = 'laplacian', gradientMode = 'conserving') / simulationState['rho'].view(-1,1)
