import torch
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState
from torch.profiler import record_function

def computeFluidDensity(fluidState, simConfig):
    with record_function("Density"):
        rho = sphOperationFluidState(fluidState, operation = 'density')
        return rho