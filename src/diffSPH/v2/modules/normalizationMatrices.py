import torch
from diffSPH.v2.math import pinv2x2
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState
from torch.profiler import record_function


def computeNormalizationMatrices(fluidState, simConfig):
    with record_function("[SPH] - Normalization Matrices (nabla x)"):
        with record_function("[SPH] - Normalization Matrices (Matrix Computation)"):
            distances = -(fluidState['fluidNeighborhood']['distances'] * fluidState['fluidNeighborhood']['supports']).view(-1,1) * fluidState['fluidNeighborhood']['vectors']
            normalizationMatrices = sphOperationFluidState(fluidState, distances, operation = 'gradient', gradientMode = 'difference')
  
        with record_function("[SPH] - Normalization Matrices (Pseudo-Inverse)"):
            L, lambdas = pinv2x2(normalizationMatrices)

        return L, normalizationMatrices, lambdas