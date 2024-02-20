import torch
from diffSPH.v2.math import pinv2x2
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState
from torch.profiler import record_function


def computeNormalizationMatrices(fluidState, simConfig):
    with record_function("Normalization"):
        with record_function("[Normalization] Gradient OP"):
            normalizationMatrices = -sphOperationFluidState(fluidState, (fluidState['fluidPositions'], fluidState['fluidPositions']), operation = 'gradient', gradientMode = 'difference')
        with record_function("[Normalization] Pinv"):
            L, lambdas = pinv2x2(normalizationMatrices)

        return L, normalizationMatrices, lambdas