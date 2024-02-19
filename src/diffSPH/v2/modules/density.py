import torch
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState

def computeFluidDensity(fluidState, simConfig):
    rho = sphOperationFluidState(fluidState, operation = 'density')
    return rho