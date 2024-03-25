import torch
from diffSPH.v2.sphOps import sphOperationStates
from torch.profiler import record_function

def computeMomentumEquation(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Momentum (rho_i nabla cdot v)"):
        return -stateA['densities'] * sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), operation = 'divergence', gradientMode='difference', neighborhood=neighborhood)
