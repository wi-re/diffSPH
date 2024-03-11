import torch
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState
from torch.profiler import record_function

def computeDivergence(fluidState, simConfig):
    with record_function("[SPH] - Fluid Divergence (nabla cdot v$)"):
        divergence = sphOperationFluidState(fluidState, (fluidState['fluidVelocities'], fluidState['fluidVelocities']), 'divergence')
        return divergence