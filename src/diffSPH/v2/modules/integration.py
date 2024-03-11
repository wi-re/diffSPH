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
    with record_function("[Simulation] - Integrate"):
        dt = config['timestep']['dt']
        scheme = config['integration']['scheme']
        assert scheme in ['semiImplicitEuler', 'explicitEuler', 'verlet', 'leapfrog', 'RK4'], f"Integration scheme {scheme} not recognized"
        dxdt, dudt, drhodt = (None, None, None)

        tempState = copy.deepcopy(perennialState)
        if scheme == 'semiImplicitEuler':
            with record_function("[Simulation] - Semi-Implicit Euler"):
                dxdt, dudt, drhodt = simulationStep(tempState, config)
                if dudt is not None:
                    perennialState['fluidVelocities'] += dudt * dt
                if drhodt is not None:
                    perennialState['fluidDensities'] += drhodt * dt
                if dxdt is not None:
                    perennialState['fluidPositions'] += dxdt * dt

            # perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt
        elif scheme == 'explicitEuler':
            with record_function("[Simulation] - Explicit Euler"):
                dxdt, dudt, drhodt = simulationStep(tempState, config)
                perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt
                if dudt is not None:
                    perennialState['fluidVelocities'] += dudt * dt
                if drhodt is not None:
                    perennialState['fluidDensities'] += drhodt * dt
                if dxdt is not None:
                    perennialState['fluidPositions'] += dxdt * dt
        elif scheme == 'verlet':
            if previousStep is None:
                with record_function("[Simulation] - Verlet (Previous Step Recomputation)"):
                    previousStep = simulationStep(perennialState, config)
            with record_function("[Simulation] - Verlet"):
                perennialState['fluidPositions'] += perennialState['fluidVelocities'] * dt + 0.5 * previousStep * dt ** 2
                dxdt, dudt, drhodt = simulationStep(tempState, config)
                if dudt is not None:
                    perennialState['fluidVelocities'] += 0.5 * (dudt + previousStep[1]) * dt
                if drhodt is not None:
                    perennialState['fluidDensities'] += 0.5 * (drhodt + previousStep[2]) * dt
                if dxdt is not None:
                    perennialState['fluidPositions'] += 0.5 * (dxdt + previousStep[0]) * dt
        elif scheme == 'leapfrog':
            if previousStep is None:
                with record_function("[Simulation] - Leapfrog (Previous Step Recomputation)"):
                    previousStep = simulationStep(tempState, config)
            with record_function("[Simulation] - Leapfrog"):
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
                tempState = copy.deepcopy(perennialState)
                dxdt, dudt, drhodt = simulationStep(tempState, config)

                # Compute the new velocity at t + dt
                if dudt is not None:
                    perennialState['fluidVelocities'] += 0.5 * dudt * dt
                if dxdt is not None:
                    perennialState['fluidPositions'] += 0.5 * dxdt * dt
                if drhodt is not None:
                    perennialState['fluidDensities'] += 0.5 * drhodt * dt
        elif scheme == 'RK4':
            with record_function("[Simulation] - RK4"):
                with record_function("[Simulation] - RK4 - k1"):
                    # Compute k1
                    dxdt_k1, dudt_k1, drhodt_k1 = simulationStep(tempState, config)

                    
                    tempState = copy.deepcopy(perennialState)
                    if dudt_k1 is not None:
                        tempState['fluidVelocities'] += dudt_k1 * dt / 2
                    if dxdt_k1 is not None:
                        tempState['fluidPositions'] += dxdt_k1 * dt / 2
                    if drhodt_k1 is not None:
                        tempState['fluidDensities'] += drhodt_k1 * dt / 2
                    # tempState['fluidPositions'] += perennialState['fluidVelocities'] * dt / 2
                with record_function("[Simulation] - RK4 - k2"):
                    # Compute k2
                    dxdt_k2, dudt_k2, drhodt_k2 = simulationStep(tempState, config)
                    tempState = copy.deepcopy(perennialState)
                    if dudt_k1 is not None:
                        tempState['fluidVelocities'] += dudt_k2 * dt / 2
                    if dxdt_k1 is not None:
                        tempState['fluidPositions'] += dxdt_k2 * dt / 2
                    if drhodt_k1 is not None:
                        tempState['fluidDensities'] += drhodt_k2 * dt / 2
                    # tempState['fluidPositions'] += (perennialState['fluidVelocities'] + dudt_k2 * dt / 2) * dt / 2
                with record_function("[Simulation] - RK4 - k3"):
                    # Compute k3
                    dxdt_k3, dudt_k3, drhodt_k3 = simulationStep(tempState, config)
                    tempState = copy.deepcopy(perennialState)
                    if dudt_k1 is not None:
                        tempState['fluidVelocities'] += dudt_k3 * dt
                    if dxdt_k1 is not None:
                        tempState['fluidPositions'] += dxdt_k3 * dt 
                    if drhodt_k1 is not None:
                        tempState['fluidDensities'] += drhodt_k3 * dt
                    # tempState['fluidPositions'] += (perennialState['fluidVelocities'] + dudt_k3 * dt / 2) * dt
                with record_function("[Simulation] - RK4 - k4"):
                    # Compute k4
                    dxdt_k4, dudt_k4, drhodt_k4 = simulationStep(tempState, config)
                with record_function("[Simulation] - RK4 - Update"):
                    # Update the position and velocity
                    if dxdt_k1 is not None:
                        dxdt = (dxdt_k1 + 2*dxdt_k2 + 2*dxdt_k3 + dxdt_k4)  / 6
                        perennialState['fluidPositions'] = perennialState['fluidPositions'] + dxdt* dt
                        # perennialState['fluidPositions'] = tempState['fluidPositions']
                    if dudt_k1 is not None:
                        dudt = (dudt_k1 + 2*dudt_k2 + 2*dudt_k3 + dudt_k4)  / 6
                        perennialState['fluidVelocities'] = perennialState['fluidVelocities'] + dudt* dt
                        # perennialState['fluidVelocities'] = tempState['fluidVelocities']
                    if drhodt_k1 is not None:
                        drhodt = (drhodt_k1 + 2*drhodt_k2 + 2*drhodt_k3 + drhodt_k4)  / 6
                        perennialState['fluidDensities'] = perennialState['fluidDensities'] + drhodt* dt
                        # perennialState['fluidDensities'] = tempState['fluidDensities']
                    # return tempState, dxdt, dudt, drhodt
                    # perennialState['fluidVelocities'] += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
                    # dudt = (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        with record_function("[Simulation] - Update Perennial State"):
            # sync perennialState with tempState   
            for k in perennialState.keys():
                if k not in ['fluidVelocities', 'fluidPositions', 'fluidDensities']:
                    perennialState[k] = tempState[k]
            if dxdt is not None:
                perennialState['fluid_dxdt'] = dxdt 
            if dudt is not None:
                perennialState['fluid_dudt'] = dudt 
            if drhodt is not None:
                perennialState['fluid_drhodt'] = drhodt 

        return perennialState, tempState, dxdt, dudt, drhodt


    
from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('integration', 'scheme', 'string', 'RK4', hint = 'The integration scheme to use. Options are: semiImplicitEuler, explicitEuler, verlet, leapfrog, RK4', required = False, export = True),
    ]