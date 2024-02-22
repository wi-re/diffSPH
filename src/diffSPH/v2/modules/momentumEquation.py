import torch
from diffSPH.v2.sphOps import sphOperationFluidState

def computeMomentumEquation(simulationState, config):
        return -simulationState['fluidDensities'] * sphOperationFluidState(simulationState, (simulationState['fluidVelocities'], simulationState['fluidVelocities']), operation = 'divergence', gradientMode='difference')
