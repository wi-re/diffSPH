import torch
from diffSPH.v2.math import pinv2x2
from diffSPH.v2.sphOps import sphOperation, sphOperationStates
from torch.profiler import record_function


def computeNormalizationMatrices(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Normalization Matrices (nabla x)"):
        with record_function("[SPH] - Normalization Matrices (Matrix Computation)"):
            distances = -(neighborhood['distances'] * neighborhood['supports']).view(-1,1) * neighborhood['vectors']
            normalizationMatrices = sphOperationStates(stateA, stateB, distances, operation = 'gradient', gradientMode = 'difference', neighborhood = neighborhood)
  
        with record_function("[SPH] - Normalization Matrices (Pseudo-Inverse)"):
            L, lambdas = pinv2x2(normalizationMatrices)

        return L, normalizationMatrices, lambdas