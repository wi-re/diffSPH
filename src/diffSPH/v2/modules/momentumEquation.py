import torch
from diffSPH.v2.sphOps import sphOperationFluidState
from torch.profiler import record_function

def computeMomentumEquation(simulationState, config):
    with record_function("[SPH] - Fluid Momentum (rho_i nabla cdot v)"):
        return -simulationState['fluidDensities'] * sphOperationFluidState(simulationState, (simulationState['fluidVelocities'], simulationState['fluidVelocities']), operation = 'divergence', gradientMode='difference')
