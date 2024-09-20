import torch
from diffSPH.v2.sphOps import sphOperation, sphOperationStates
from torch.profiler import record_function


def computeDensity(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Density"):
        rho = sphOperationStates(
            stateA,
            stateB,
            quantities=None,
            operation="density",
            neighborhood=neighborhood,
        )
        return rho
