import torch
from diffSPH.parameter import Parameter
from torch.profiler import record_function

def computeCFL(simulationState, config):
    with record_function("CFL"):
        dt = simulationState['dt']
        max_velocity = torch.max(simulationState['fluidVelocities'].norm(dim = 1))
        h = simulationState['fluidSupports'].min()
        return config['integration']['CFL'] * h / (max_velocity / dt)

