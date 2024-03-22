import torch
from diffSPH.v2.sphOps import sphOperation, sphOperationStates
from torch.profiler import record_function

def computeDivergence(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Fluid Divergence (nabla cdot v$)"):
        divergence = sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), neighborhood=neighborhood, operation='divergence')
        return divergence