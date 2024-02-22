import torch
from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices

def computeDensityDiffusion(simulationState, config):
    L, normalizationMatrices, lambdas = computeNormalizationMatrices(simulationState, config)


    normalizedGradient = torch.matmul(normalizationMatrices[i], simulationState['fluidNeighborhood']['gradients'].view(-1,1))