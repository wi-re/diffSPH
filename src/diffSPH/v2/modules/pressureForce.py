import torch
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum

from torch.profiler import record_function

def computePressureAccel(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Pressure Acceleration (1/rho nabla p)"):
        return -sphOperationStates(stateA, stateB, (stateA['pressures'], stateB['pressures']), operation = 'gradient', gradientMode='summation', neighborhood= neighborhood) / stateA['densities'].view(-1,1)
    

def computePressureAccelSwitch(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Pressure Acceleration (1/rho nabla p) [Antuono Switch]"):
        (i,j) = neighborhood['indices']
        p_i = stateA['pressures'][i]
        p_j = stateB['pressures'][j]
        p_ij = torch.where(torch.logical_or(p_i >= 0, stateA['surfaceMask'][i] > 0.5), p_j + p_i, p_j - p_i)
        p_ij = p_j + p_i

        return -sphOperationStates(stateA, stateB, p_ij, operation = 'gradient', gradientMode='summation') / stateA['densities'].view(-1,1)