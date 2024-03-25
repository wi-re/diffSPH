import torch
from diffSPH.parameter import Parameter
from torch.profiler import record_function

def computeCFL(dt, state, config):
    with record_function("CFL"):
        # dt = state['dt']
        max_velocity = torch.max(state['velocities'].norm(dim = 1))
        h = state['supports'].min()
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
                    perennialState['fluid']['velocities'] += dudt * dt
                if drhodt is not None:
                    perennialState['fluid']['densities'] += drhodt * dt
                if dxdt is not None:
                    perennialState['fluid']['positions'] += dxdt * dt

            # perennialState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt
        elif scheme == 'explicitEuler':
            with record_function("[Simulation] - Explicit Euler"):
                dxdt, dudt, drhodt = simulationStep(tempState, config)
                perennialState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt
                if dudt is not None:
                    perennialState['fluid']['velocities'] += dudt * dt
                if drhodt is not None:
                    perennialState['fluid']['densities'] += drhodt * dt
                if dxdt is not None:
                    perennialState['fluid']['positions'] += dxdt * dt
        elif scheme == 'verlet':
            if previousStep is None:
                with record_function("[Simulation] - Verlet (Previous Step Recomputation)"):
                    previousStep = simulationStep(perennialState, config)
            with record_function("[Simulation] - Verlet"):
                perennialState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt + 0.5 * previousStep * dt ** 2
                dxdt, dudt, drhodt = simulationStep(tempState, config)
                if dudt is not None:
                    perennialState['fluid']['velocities'] += 0.5 * (dudt + previousStep[1]) * dt
                if drhodt is not None:
                    perennialState['fluid']['densities'] += 0.5 * (drhodt + previousStep[2]) * dt
                if dxdt is not None:
                    perennialState['fluid']['positions'] += 0.5 * (dxdt + previousStep[0]) * dt
        elif scheme == 'leapfrog':
            if previousStep is None:
                with record_function("[Simulation] - Leapfrog (Previous Step Recomputation)"):
                    previousStep = simulationStep(tempState, config)
            with record_function("[Simulation] - Leapfrog"):
                # Compute the new velocity at t + dt/2
                if previousStep[0] is not None:
                    perennialState['fluid']['positions'] += 0.5 * previousStep[0] * dt
                if previousStep[1] is not None:
                    perennialState['fluid']['velocities'] += 0.5 * previousStep[1] * dt
                if previousStep[2] is not None:
                    perennialState['fluid']['densities'] += 0.5 * previousStep[2] * dt

                # Compute the new position at t + dt
                perennialState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt

                # Compute the new acceleration at t + dt
                tempState = copy.deepcopy(perennialState)
                dxdt, dudt, drhodt = simulationStep(tempState, config)

                # Compute the new velocity at t + dt
                if dudt is not None:
                    perennialState['fluid']['velocities'] += 0.5 * dudt * dt
                if dxdt is not None:
                    perennialState['fluid']['positions'] += 0.5 * dxdt * dt
                if drhodt is not None:
                    perennialState['fluid']['densities'] += 0.5 * drhodt * dt
        elif scheme == 'RK4':
            with record_function("[Simulation] - RK4"):
                with record_function("[Simulation] - RK4 - k1"):
                    # Compute k1
                    dxdt_k1, dudt_k1, drhodt_k1 = simulationStep(tempState, config)

                    
                    tempState = copy.deepcopy(perennialState)
                    if dudt_k1 is not None:
                        tempState['fluid']['velocities'] += dudt_k1 * dt / 2
                    if dxdt_k1 is not None:
                        tempState['fluid']['positions'] += dxdt_k1 * dt / 2
                    if drhodt_k1 is not None:
                        tempState['fluid']['densities'] += drhodt_k1 * dt / 2
                    # tempState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt / 2
                with record_function("[Simulation] - RK4 - k2"):
                    # Compute k2
                    dxdt_k2, dudt_k2, drhodt_k2 = simulationStep(tempState, config)
                    tempState = copy.deepcopy(perennialState)
                    if dudt_k1 is not None:
                        tempState['fluid']['velocities'] += dudt_k2 * dt / 2
                    if dxdt_k1 is not None:
                        tempState['fluid']['positions'] += dxdt_k2 * dt / 2
                    if drhodt_k1 is not None:
                        tempState['fluid']['densities'] += drhodt_k2 * dt / 2
                    # tempState['fluid']['positions'] += (perennialState['fluid']['velocities'] + dudt_k2 * dt / 2) * dt / 2
                with record_function("[Simulation] - RK4 - k3"):
                    # Compute k3
                    dxdt_k3, dudt_k3, drhodt_k3 = simulationStep(tempState, config)
                    tempState = copy.deepcopy(perennialState)
                    if dudt_k1 is not None:
                        tempState['fluid']['velocities'] += dudt_k3 * dt
                    if dxdt_k1 is not None:
                        tempState['fluid']['positions'] += dxdt_k3 * dt 
                    if drhodt_k1 is not None:
                        tempState['fluid']['densities'] += drhodt_k3 * dt
                    # tempState['fluid']['positions'] += (perennialState['fluid']['velocities'] + dudt_k3 * dt / 2) * dt
                with record_function("[Simulation] - RK4 - k4"):
                    # Compute k4
                    dxdt_k4, dudt_k4, drhodt_k4 = simulationStep(tempState, config)
                with record_function("[Simulation] - RK4 - Update"):
                    # Update the position and velocity
                    if dxdt_k1 is not None:
                        dxdt = (dxdt_k1 + 2*dxdt_k2 + 2*dxdt_k3 + dxdt_k4)  / 6
                        perennialState['fluid']['positions'] = perennialState['fluid']['positions'] + dxdt* dt
                        # perennialState['fluid']['positions'] = tempState['fluid']['positions']
                    if dudt_k1 is not None:
                        dudt = (dudt_k1 + 2*dudt_k2 + 2*dudt_k3 + dudt_k4)  / 6
                        perennialState['fluid']['velocities'] = perennialState['fluid']['velocities'] + dudt* dt
                        # perennialState['fluid']['velocities'] = tempState['fluid']['velocities']
                    if drhodt_k1 is not None:
                        drhodt = (drhodt_k1 + 2*drhodt_k2 + 2*drhodt_k3 + drhodt_k4)  / 6
                        perennialState['fluid']['densities'] = perennialState['fluid']['densities'] + drhodt* dt
                        # perennialState['fluid']['densities'] = tempState['fluid']['densities']
                    # return tempState, dxdt, dudt, drhodt
                    # perennialState['fluid']['velocities'] += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
                    # dudt = (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        with record_function("[Simulation] - Update Perennial State"):
            # sync perennialState with tempState   
            for k in perennialState.keys():
                if not isinstance(perennialState[k], dict):
                    continue
                for kk in perennialState[k].keys():
                    if kk not in ['velocities', 'positions', 'densities']:
                        temp = perennialState[k][kk]
                        perennialState[k][kk] = tempState[k][kk]
                        tempState[k][kk] = temp
                # if k not in ['fluid']['velocities', 'fluid']['positions', 'fluid']['densities']:
                #     temp = perennialState[k]
                #     perennialState[k] = tempState[k]
                #     tempState[k] = temp
            if dxdt is not None:
                perennialState['fluid']['dxdt'] = dxdt 
            if dudt is not None:
                perennialState['fluid']['dudt'] = dudt 
            if drhodt is not None:
                perennialState['fluid']['drhodt'] = drhodt 
            if 'neighborhood' in tempState['fluid']:
                del tempState['fluid']['neighborhood']
            if 'neighborhood' in tempState['boundary']:
                del tempState['boundary']['neighborhood']
            if 'neighborhood' in tempState:
                del tempState['neighborhood']

        return perennialState, tempState, dxdt, dudt, drhodt


    
from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('integration', 'scheme', 'string', 'RK4', hint = 'The integration scheme to use. Options are: semiImplicitEuler, explicitEuler, verlet, leapfrog, RK4', required = False, export = True),
    ]