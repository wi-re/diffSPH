import torch
from diffSPH.v2.math import pinv2x2
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState


def computeNormalizationMatrices(fluidState, simConfig):
    normalizationMatrices = -sphOperationFluidState(fluidState, (fluidState['fluidPositions'], fluidState['fluidPositions']), operation = 'gradient', gradientMode = 'difference')

    L, lambdas = pinv2x2(normalizationMatrices)

    return L, normalizationMatrices, lambdas