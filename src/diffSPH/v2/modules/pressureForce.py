import torch
from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum

from torch.profiler import record_function

def computePressureAccel(simulationState, config):
    with record_function("PressureAccel"):
        return -sphOperationFluidState(simulationState, (simulationState['fluidPressures'], simulationState['fluidPressures']), operation = 'gradient', gradientMode='summation') / simulationState['fluidDensities'].view(-1,1)
    

def computePressureAccelSwitch(simulationState, config):
    with record_function("PressureAccel"):
        (i,j) = simulationState['fluidNeighborhood']['indices']
        p_i = simulationState['fluidPressures'][i]
        p_j = simulationState['fluidPressures'][j]
        p_ij = torch.where(torch.logical_or(p_i >= 0, simulationState['fluidSurfaceMask'][i] > 0.5), p_j + p_i, p_j - p_i)

        return -sphOperationFluidState(simulationState, p_ij, operation = 'gradient', gradientMode='summation') / simulationState['fluidDensities'].view(-1,1)