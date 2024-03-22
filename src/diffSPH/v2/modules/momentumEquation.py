import torch
from diffSPH.v2.sphOps import sphOperationStates
from torch.profiler import record_function

def computeMomentumEquation(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Momentum (rho_i nabla cdot v)"):
        return -stateA['fluidDensities'] * sphOperationStates(stateA, stateB, (stateA['fluidVelocities'], stateB['fluidVelocities']), operation = 'divergence', gradientMode='difference', neighborhood=neighborhood)
