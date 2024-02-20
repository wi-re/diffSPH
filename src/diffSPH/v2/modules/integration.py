import torch
from diffSPH.parameter import Parameter
from torch.profiler import record_function

def computeCFL(simulationState, config):
    with record_function("CFL"):
        dt = simulationState['dt']
        max_velocity = torch.max(simulationState['fluidVelocities'].norm(dim = 1))
        h = simulationState['fluidSupports'].min()
        return config['integration']['CFL'] * h / (max_velocity / dt)
    
import copy
def integrate(simulationStep, perennialState, config, previousStep = None):
    dt = config['integration']['dt']
    scheme = config['integration']['scheme']
    assert scheme in ['semiImplicitEuler', 'explicitEuler', 'verlet', 'leapfrog', 'RK4'], f"Integration scheme {scheme} not recognized"

    if scheme == 'semiImplicitEuler':
        dudt = simulationStep(perennialState, config)
        perennialState['fluidVelocities'] += dudt * dt
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt
    elif scheme == 'explicitEuler':
        dudt = simulationStep(perennialState, config)
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt
        perennialState['fluidVelocities'] += dudt * dt
    elif scheme == 'verlet':
        if previousStep is None:
            previousStep = simulationStep(perennialState, config)
        
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt + 0.5 * previousStep * dt ** 2
        dudt = simulationStep(perennialState, config)
        perennialState['fluidVelocities'] += 0.5 * (dudt + previousStep) * dt
    elif scheme == 'leapfrog':
        if previousStep is None:
            previousStep = simulationStep(perennialState, config)

        # Compute the new velocity at t + dt/2
        perennialState['fluidVelocities'] += 0.5 * previousStep * dt

        # Compute the new position at t + dt
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt

        # Compute the new acceleration at t + dt
        dudt = simulationStep(perennialState, config)

        # Compute the new velocity at t + dt
        perennialState['fluidVelocities'] += 0.5 * dudt * dt
    elif scheme == 'RK4':
        # Compute k1
        k1 = simulationStep(perennialState, config)
        tempState = copy.deepcopy(perennialState)
        tempState['fluidVelocities'] += k1 * dt / 2
        tempState['fluidPositions'] += perennialState['fluidVelocities'] * dt / 2

        # Compute k2
        k2 = simulationStep(tempState, config)
        tempState = copy.deepcopy(perennialState)
        tempState['fluidVelocities'] += k2 * dt / 2
        tempState['fluidPositions'] += (perennialState['fluidVelocities'] + k1 * dt / 2) * dt / 2

        # Compute k3
        k3 = simulationStep(tempState, config)
        tempState = copy.deepcopy(perennialState)
        tempState['fluidVelocities'] += k3 * dt
        tempState['fluidPositions'] += (perennialState['fluidVelocities'] + k2 * dt / 2) * dt

        # Compute k4
        k4 = simulationStep(tempState, config)

        # Update the position and velocity
        perennialState['fluidPositions'] += dt * (perennialState['fluidVelocities'] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6)
        perennialState['fluidVelocities'] += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        dudt = (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return perennialState, dudt


    
from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('integration', 'dt', float, 0.01, required = False, export = True),
    Parameter('integration', 'adaptiveTimestep', bool, False, required = False, export = True),
    Parameter('integration', 'CFL', float, 0.25, required = False, export = True),
    Parameter('integration', 'maxDt', float, 0.1, required = False, export = True),
    Parameter('integration', 'minDt', float, 0.001, required = False, export = True),
    Parameter('integration', 'scheme', 'string', 'semiImplicitEuler', hint = 'The integration scheme to use. Options are: semiImplicitEuler, explicitEuler, verlet, leapfrog, RK4', required = False, export = True),
    ]