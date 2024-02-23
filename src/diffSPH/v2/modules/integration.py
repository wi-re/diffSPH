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
    dxdt, dudt, drhodt = (None, None, None)

    if scheme == 'semiImplicitEuler':
        dxdt, dudt, drhodt = simulationStep(perennialState, config)
        if dudt is not None:
            perennialState['fluidVelocities'] += dudt * dt
        if drhodt is not None:
            perennialState['fluidDensities'] += drhodt * dt
        if dxdt is not None:
            perennialState['fluidPositions'] += dxdt * dt

        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt
    elif scheme == 'explicitEuler':
        dxdt, dudt, drhodt = simulationStep(perennialState, config)
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt
        if dudt is not None:
            perennialState['fluidVelocities'] += dudt * dt
        if drhodt is not None:
            perennialState['fluidDensities'] += drhodt * dt
        if dxdt is not None:
            perennialState['fluidPositions'] += dxdt * dt
    elif scheme == 'verlet':
        if previousStep is None:
            previousStep = simulationStep(perennialState, config)
        
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt + 0.5 * previousStep * dt ** 2
        dxdt, dudt, drhodt = simulationStep(perennialState, config)
        if dudt is not None:
            perennialState['fluidVelocities'] += 0.5 * (dudt + previousStep[1]) * dt
        if drhodt is not None:
            perennialState['fluidDensities'] += 0.5 * (drhodt + previousStep[2]) * dt
        if dxdt is not None:
            perennialState['fluidPositions'] += 0.5 * (dxdt + previousStep[0]) * dt
    elif scheme == 'leapfrog':
        if previousStep is None:
            previousStep = simulationStep(perennialState, config)

        # Compute the new velocity at t + dt/2
        if previousStep[0] is not None:
            perennialState['fluidPositions'] += 0.5 * previousStep[0] * dt
        if previousStep[1] is not None:
            perennialState['fluidVelocities'] += 0.5 * previousStep[1] * dt
        if previousStep[2] is not None:
            perennialState['fluidDensities'] += 0.5 * previousStep[2] * dt

        # Compute the new position at t + dt
        perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt

        # Compute the new acceleration at t + dt
        dxdt, dudt, drhodt = simulationStep(perennialState, config)

        # Compute the new velocity at t + dt
        if dudt is not None:
            perennialState['fluidVelocities'] += 0.5 * dudt * dt
        if dxdt is not None:
            perennialState['fluidPositions'] += 0.5 * dxdt * dt
        if drhodt is not None:
            perennialState['fluidDensities'] += 0.5 * drhodt * dt
    elif scheme == 'RK4':
        # Compute k1
        dxdt_k1, dudt_k1, drhodt_k1 = simulationStep(perennialState, config)
        tempState = copy.deepcopy(perennialState)
        if dudt_k1 is not None:
            tempState['fluidVelocities'] += dudt_k1 * dt / 2
        if dxdt_k1 is not None:
            tempState['fluidPositions'] += dxdt_k1 * dt / 2
        if drhodt_k1 is not None:
            tempState['fluidDensities'] += drhodt_k1 * dt / 2
        tempState['fluidPositions'] += perennialState['fluidVelocities'] * dt / 2

        # Compute k2
        dxdt_k2, dudt_k2, drhodt_k2 = simulationStep(tempState, config)
        tempState = copy.deepcopy(perennialState)
        if dudt_k1 is not None:
            tempState['fluidVelocities'] += dudt_k2 * dt / 2
        if dxdt_k1 is not None:
            tempState['fluidPositions'] += dxdt_k2 * dt / 2
        if drhodt_k1 is not None:
            tempState['fluidDensities'] += drhodt_k2 * dt / 2
        tempState['fluidPositions'] += (perennialState['fluidVelocities'] + dudt_k2 * dt / 2) * dt / 2

        # Compute k3
        dxdt_k3, dudt_k3, drhodt_k3 = simulationStep(tempState, config)
        tempState = copy.deepcopy(perennialState)
        if dudt_k1 is not None:
            tempState['fluidVelocities'] += dudt_k3 * dt / 2
        if dxdt_k1 is not None:
            tempState['fluidPositions'] += dxdt_k3 * dt / 2
        if drhodt_k1 is not None:
            tempState['fluidDensities'] += drhodt_k3 * dt / 2
        tempState['fluidPositions'] += (perennialState['fluidVelocities'] + dudt_k3 * dt / 2) * dt

        # Compute k4
        dxdt_k4, dudt_k4, drhodt_k4 = simulationStep(tempState, config)

        # Update the position and velocity
        if dudt_k1 is not None:
            perennialState['fluidPositions'] += dt * (perennialState['fluidVelocities'] + (dudt_k1 + 2*dudt_k2 + 2*dudt_k3 + dudt_k4) * dt / 6)
            perennialState['fluidVelocities'] += (dudt_k1 + 2*dudt_k2 + 2*dudt_k3 + dudt_k4) * dt / 6
            dudt = (dudt_k1 + 2*dudt_k2 + 2*dudt_k3 + dudt_k4) * dt / 6
        if drhodt_k1 is not None:
            perennialState['fluidDensities'] += (drhodt_k1 + 2*drhodt_k2 + 2*drhodt_k3 + drhodt_k4) * dt / 6
            drhodt = (drhodt_k1 + 2*drhodt_k2 + 2*drhodt_k3 + drhodt_k4) * dt / 6
        if dxdt_k1 is not None:
            perennialState['fluidPositions'] += (dxdt_k1 + 2*dxdt_k2 + 2*dxdt_k3 + dxdt_k4) * dt / 6
            dxdt = (dxdt_k1 + 2*dxdt_k2 + 2*dxdt_k3 + dxdt_k4) * dt / 6

        # perennialState['fluidVelocities'] += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        # dudt = (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return perennialState, dxdt, dudt, drhodt


    
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