import torch
from diffSPH.parameter import Parameter
from torch.profiler import record_function

def computeCFL(dt, state, config):
    with record_function("CFL"):
        # dt = state['dt']
        max_velocity = torch.max(state['velocities'].norm(dim = 1))
        h = state['supports'].min()
        return config['integration']['CFL'] * h / (max_velocity / dt)
    
def update(dt, update, state):
    if update is not None:
        dxdt, dudt, drhodt = update
        if dxdt is not None:
            state['positions'] += dxdt * dt
        if dudt is not None:
            state['velocities'] += dudt * dt
        if drhodt is not None:
            state['densities'] += drhodt * dt
        
def updateStates(dt, update, stateA, stateB):
    updateA, updateB = update
    if updateA is not None:
        dxdt, dudt, drhodt = updateA
        if dxdt is not None:
            stateA['positions'] += dxdt * dt
        if dudt is not None:
            stateA['velocities'] += dudt * dt
        if drhodt is not None:
            stateA['densities'] += drhodt * dt
    if updateB is not None:
        dxdt, dudt, drhodt = updateB
        if dxdt is not None:
            stateB['positions'] += dxdt * dt
        if dudt is not None:
            stateB['velocities'] += dudt * dt
        if drhodt is not None:
            stateB['densities'] += drhodt * dt

def createTempState(perennialState, tempState_prior = None):
    # neighborhood

    tempState = copy.deepcopy(perennialState)
    if tempState_prior is not None:
        if 'neighborhood' in tempState_prior['fluid']:
            tempState['fluid']['neighborhood'] = tempState_prior['fluid']['neighborhood']
        if 'datastructure' in tempState_prior['fluid']:
            tempState['fluid']['datastructure'] = tempState_prior['fluid']['datastructure']
        if 'boundary'in tempState:

            if 'neighborhood' in tempState_prior['boundary']:
                tempState['boundary']['neighborhood'] = tempState_prior['boundary']['neighborhood']
            if 'datastructure' in tempState_prior['boundary']:
                tempState['boundary']['datastructure'] = tempState_prior['boundary']['datastructure']

            if 'boundaryToFluidNeighborhood' in tempState_prior:
                tempState['boundaryToFluidNeighborhood'] = tempState_prior['boundaryToFluidNeighborhood']
            if 'fluidToBoundaryNeighborhood' in tempState_prior:
                tempState['fluidToBoundaryNeighborhood'] = tempState_prior['fluidToBoundaryNeighborhood']
            if 'boundaryGhostToFluidNeighborhood' in tempState_prior:
                tempState['boundaryGhostToFluidNeighborhood'] = tempState_prior['boundaryGhostToFluidNeighborhood']


        del tempState_prior
    # Clean up residual neighborhoods in perennial State
    if 'neighborhood' in perennialState['fluid']:
        del perennialState['fluid']['neighborhood']
    if 'datastructure' in perennialState['fluid']:
        del perennialState['fluid']['neighborhood']
    if 'boundary' in perennialState:
        if 'neighborhood' in perennialState['boundary']:
            del perennialState['boundary']['neighborhood']
        if 'datastructure' in perennialState['boundary']:
            del perennialState['boundary']['datastructure']
        if 'boundaryToFluidNeighborhood' in perennialState:
            del perennialState['boundaryToFluidNeighborhood']
        if 'fluidToBoundaryNeighborhood'in perennialState:
            del perennialState['fluidToBoundaryNeighborhood']
        if 'boundaryGhostToFluidNeighborhood'in perennialState:
            del perennialState['boundaryGhostToFluidNeighborhood']

        # boundaryToFluidNeighborhood']fluidToBoundaryNeighborhood
    return tempState


import copy
def integrate(simulationStep, perennialState, config, previousStep = None):
    with record_function("[Simulation] - Integrate"):
        dt = config['timestep']['dt']
        scheme = config['integration']['scheme']
        assert scheme in ['semiImplicitEuler', 'explicitEuler', 'verlet', 'leapfrog', 'RK4'], f"Integration scheme {scheme} not recognized"
        fluidUpdate, boundaryUpdate = None, None

        tempState = createTempState(perennialState, None)
        # tempState = copy.deepcopy(perennialState)
        # if 'neighborhood' in perennialState['fluid']:
        #     del perennialState['fluid']['neighborhood']
        # if 'datastructure' in perennialState['fluid']:
        #     del perennialState['fluid']['neighborhood']
        # if 'neighborhood' in perennialState['boundary']:
        #     del perennialState['boundary']['neighborhood']
        # if 'datastructure' in perennialState['boundary']:
        #     del perennialState['boundary']['datastructure']


        if scheme == 'semiImplicitEuler':
            with record_function("[Simulation] - Semi-Implicit Euler"):
                fluidUpdate, boundaryUpdate = simulationStep(tempState, config)
                if fluidUpdate[1] is not None:
                    perennialState['fluid']['velocities'] += fluidUpdate[1] * dt                
                if config['boundary']['active']:
                    if boundaryUpdate[1] is not None:
                        perennialState['boundary']['velocities'] += boundaryUpdate[1] * dt
                dxdt, dudt, drhodt = fluidUpdate
                updateStates(dt, (fluidUpdate, boundaryUpdate), perennialState['fluid'], perennialState['boundary'] if 'boundary' in perennialState else None)

        elif scheme == 'explicitEuler':
            with record_function("[Simulation] - Explicit Euler"):
                fluidUpdate, boundaryUpdate = simulationStep(tempState, config)
                updateStates(dt, (fluidUpdate, boundaryUpdate), perennialState['fluid'], perennialState['boundary'] if 'boundary' in perennialState else None)
        elif scheme == 'verlet':
            if previousStep is None:
                with record_function("[Simulation] - Verlet (Previous Step Recomputation)"):
                    previousStep = simulationStep(perennialState, config)
            with record_function("[Simulation] - Verlet"):
                if previousStep is not None:
                    perennialState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt + 0.5 * previousStep[0][1] * dt ** 2
                if config['boundary']['active']:
                    if previousStep is not None:
                        perennialState['boundary']['positions'] += perennialState['boundary']['velocities'] * dt + 0.5 * previousStep[1][1] * dt ** 2

                fluidUpdate, boundaryUpdate = simulationStep(tempState, config)
                if fluidUpdate is not None:
                    dxdt, dudt, drhodt = fluidUpdate
                    if dudt is not None:
                        perennialState['fluid']['velocities'] += 0.5 * (dudt + previousStep[0][1]) * dt
                    if drhodt is not None:
                        perennialState['fluid']['densities'] += 0.5 * (drhodt + previousStep[0][2]) * dt
                    if dxdt is not None:
                        perennialState['fluid']['positions'] += 0.5 * (dxdt + previousStep[0][0]) * dt
                if config['boundary']['active']:
                    dxdt, dudt, drhodt = boundaryUpdate
                    if dudt is not None:
                        perennialState['boundary']['velocities'] += 0.5 * (dudt + previousStep[1][1]) * dt
                    if drhodt is not None:
                        perennialState['boundary']['densities'] += 0.5 * (drhodt + previousStep[1][2]) * dt
                    if dxdt is not None:
                        perennialState['boundary']['positions'] += 0.5 * (dxdt + previousStep[1][0]) * dt

        elif scheme == 'leapfrog':
            if previousStep is None:
                with record_function("[Simulation] - Leapfrog (Previous Step Recomputation)"):
                    previousStep = simulationStep(tempState, config)
            with record_function("[Simulation] - Leapfrog"):
                # Compute the new velocity at t + dt/2
                if previousStep[0] is not None:
                    if previousStep[0][0] is not None:
                        perennialState['fluid']['positions'] += 0.5 * previousStep[0][0] * dt
                    if previousStep[0][1] is not None:
                        perennialState['fluid']['velocities'] += 0.5 * previousStep[0][1] * dt
                    if previousStep[0][2] is not None:
                        perennialState['fluid']['densities'] += 0.5 * previousStep[0][2] * dt
                if config['boundary']['active']:
                    if previousStep[1] is not None:
                        if previousStep[1][0] is not None:
                            perennialState['boundary']['positions'] += 0.5 * previousStep[1][0] * dt
                        if previousStep[1][1] is not None:
                            perennialState['boundary']['velocities'] += 0.5 * previousStep[1][1] * dt
                        if previousStep[1][2] is not None:
                            perennialState['boundary']['densities'] += 0.5 * previousStep[1][2] * dt

                # Compute the new position at t + dt
                perennialState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt
                if config['boundary']['active']:
                    perennialState['boundary']['positions'] += perennialState['boundary']['velocities'] * dt

                # Compute the new acceleration at t + dt
                tempState = copy.deepcopy(perennialState)
                fluidUpdate, boundaryUpdate = simulationStep(tempState, config)

                # Compute the new velocity at t + dt
                if fluidUpdate is not None:
                    dxdt, dudt, drhodt = fluidUpdate
                    if dudt is not None:
                        perennialState['fluid']['velocities'] += 0.5 * dudt * dt
                    if dxdt is not None:
                        perennialState['fluid']['positions'] += 0.5 * dxdt * dt
                    if drhodt is not None:
                        perennialState['fluid']['densities'] += 0.5 * drhodt * dt
                if config['boundary']['active']:
                    if boundaryUpdate is not None:
                        dxdt, dudt, drhodt = boundaryUpdate
                        if dudt is not None:
                            perennialState['boundary']['velocities'] += 0.5 * dudt * dt
                        if dxdt is not None:
                            perennialState['boundary']['positions'] += 0.5 * dxdt * dt
                        if drhodt is not None:
                            perennialState['boundary']['densities'] += 0.5 * drhodt * dt
        elif scheme == 'RK4':
            with record_function("[Simulation] - RK4"):
                with record_function("[Simulation] - RK4 - k1"):
                    # print("RK4 - k1")
                    # Compute k1
                    fluidUpdate_k1, boundaryUpdate_k1 = simulationStep(tempState, config)

                    tempState = createTempState(perennialState, tempState)
                    # neighborhood = tempState['fluid']['neighborhood'] if 'neighborhood' in tempState['fluid'] else None
                    # datastructure = tempState['fluid']['datastructure'] if 'datastructure' in tempState['fluid'] else None
                    # tempState = copy.deepcopy(perennialState)
                    # tempState['fluid']['neighborhood'] = neighborhood
                    # tempState['fluid']['datastructure'] = datastructure
                    # del neighborhood
                    # del datastructure
                    updateStates(dt / 2, (fluidUpdate_k1, boundaryUpdate_k1), tempState['fluid'], tempState['boundary'] if 'boundary' in tempState else None)
                    # tempState['fluid']['positions'] += perennialState['fluid']['velocities'] * dt / 2
                with record_function("[Simulation] - RK4 - k2"):
                    # print("RK4 - k2")
                    # Compute k2
                    fluidUpdate_k2, boundaryUpdate_k2 = simulationStep(tempState, config)

                    tempState = createTempState(perennialState, tempState)
                    # neighborhood = tempState['fluid']['neighborhood'] if 'neighborhood' in tempState['fluid'] else None
                    # tempState = copy.deepcopy(perennialState)
                    # tempState['fluid']['neighborhood'] = neighborhood
                    # del neighborhood
                    updateStates(dt / 2, (fluidUpdate_k2, boundaryUpdate_k2), tempState['fluid'], tempState['boundary'] if 'boundary' in tempState else None)
                    # tempState['fluid']['positions'] += (perennialState['fluid']['velocities'] + dudt_k2 * dt / 2) * dt / 2
                with record_function("[Simulation] - RK4 - k3"):
                    # print("RK4 - k3")
                    # Compute k3
                    fluidUpdate_k3, boundaryUpdate_k3 = simulationStep(tempState, config)
                    tempState = createTempState(perennialState, tempState)
                    # neighborhood = tempState['fluid']['neighborhood'] if 'neighborhood' in tempState['fluid'] else None
                    # tempState = copy.deepcopy(perennialState)
                    # tempState['fluid']['neighborhood'] = neighborhood
                    # del neighborhood
                    updateStates(dt, (fluidUpdate_k3, boundaryUpdate_k3), tempState['fluid'], tempState['boundary'] if 'boundary' in tempState else None)
                    # tempState['fluid']['positions'] += (perennialState['fluid']['velocities'] + dudt_k3 * dt / 2) * dt
                with record_function("[Simulation] - RK4 - k4"):
                    # print("RK4 - k4")
                    # Compute k4
                    fluidUpdate_k4, boundaryUpdate_k4 = simulationStep(tempState, config)
                with record_function("[Simulation] - RK4 - Update"):
                    # print("RK4 - update")
                    # Update the position and velocity
                    dxdt, dudt, drhodt = None, None, None
                    if fluidUpdate_k1[0] is not None:
                        dxdt = (fluidUpdate_k1[0] + 2 * fluidUpdate_k2[0] + 2 * fluidUpdate_k3[0] + fluidUpdate_k4[0]) / 6
                        perennialState['fluid']['positions'] += dxdt * dt
                    if fluidUpdate_k1[1] is not None:
                        dudt = (fluidUpdate_k1[1] + 2 * fluidUpdate_k2[1] + 2 * fluidUpdate_k3[1] + fluidUpdate_k4[1]) / 6
                        perennialState['fluid']['velocities'] += dudt * dt
                    if fluidUpdate_k1[2] is not None:
                        drhodt = (fluidUpdate_k1[2] + 2 * fluidUpdate_k2[2] + 2 * fluidUpdate_k3[2] + fluidUpdate_k4[2]) / 6
                        perennialState['fluid']['densities'] += drhodt * dt
                    fluidUpdate = (dxdt, dudt, drhodt)
                    if config['boundary']['active']:
                        dxdt, dudt, drhodt = None, None, None
                        if boundaryUpdate_k1[0] is not None:
                            dxdt = (boundaryUpdate_k1[0] + 2 * boundaryUpdate_k2[0] + 2 * boundaryUpdate_k3[0] + boundaryUpdate_k4[0]) / 6
                            perennialState['boundary']['positions'] += dxdt * dt
                        if boundaryUpdate_k1[1] is not None:
                            dudt = (boundaryUpdate_k1[1] + 2 * boundaryUpdate_k2[1] + 2 * boundaryUpdate_k3[1] + boundaryUpdate_k4[1]) / 6
                            perennialState['boundary']['velocities'] += dudt * dt
                        if boundaryUpdate_k1[2] is not None:
                            drhodt = (boundaryUpdate_k1[2] + 2 * boundaryUpdate_k2[2] + 2 * boundaryUpdate_k3[2] + boundaryUpdate_k4[2]) / 6
                            perennialState['boundary']['densities'] += drhodt * dt
                        boundaryUpdate = (dxdt, dudt, drhodt)
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
            if 'neighborhood' in tempState['fluid']:
                perennialState['fluid']['neighborhood'] = tempState['fluid']['neighborhood']
                del tempState['fluid']['neighborhood']
            if 'datastructure' in tempState['fluid']:
                perennialState['fluid']['datastructure'] = tempState['fluid']['datastructure']
                del tempState['fluid']['datastructure']
            if 'boundary' in tempState:
                if 'neighborhood' in tempState['boundary']:
                    perennialState['boundary']['neighborhood'] = tempState['boundary']['neighborhood']
                    del tempState['boundary']['neighborhood']
                if 'datastructure' in tempState['boundary']:
                    perennialState['boundary']['datastructure'] = tempState['boundary']['datastructure']
                    del tempState['boundary']['datastructure']
                        
                if 'boundaryToFluidNeighborhood' in tempState:
                    perennialState['boundaryToFluidNeighborhood'] = tempState['boundaryToFluidNeighborhood']
                    del tempState['boundaryToFluidNeighborhood']
                if 'fluidToBoundaryNeighborhood' in tempState:
                    perennialState['fluidToBoundaryNeighborhood'] = tempState['fluidToBoundaryNeighborhood']
                    del tempState['fluidToBoundaryNeighborhood']
                if 'boundaryGhostToFluidNeighborhood' in tempState:
                    perennialState['boundaryGhostToFluidNeighborhood'] = tempState['boundaryGhostToFluidNeighborhood']
                    del tempState['boundaryGhostToFluidNeighborhood']



            if fluidUpdate[0] is not None:
                perennialState['fluid']['dxdt'] = fluidUpdate[0] 
            if fluidUpdate[1] is not None:
                perennialState['fluid']['dudt'] = fluidUpdate[1] 
            if fluidUpdate[2] is not None:
                perennialState['fluid']['drhodt'] = fluidUpdate[2] 
            if 'boundary' in perennialState:
                if boundaryUpdate is not None:
                    if boundaryUpdate[0] is not None:
                        perennialState['boundary']['dxdt'] = boundaryUpdate[0] 
                    if boundaryUpdate[1] is not None:
                        perennialState['boundary']['dudt'] = boundaryUpdate[1] 
                    if boundaryUpdate[2] is not None:
                        perennialState['boundary']['drhodt'] = boundaryUpdate[2]
            # if 'neighborhood' in tempState['fluid']:
                # del tempState['fluid']['neighborhood']
            if 'boundary' in tempState and 'neighborhood' in tempState['boundary']:
                # del tempState['boundary']['neighborhood']
                perennialState['boundary']['densities'] = tempState['boundary']['densities']
            # if 'neighborhood' in tempState:
                # del tempState['neighborhood']
        # print("RK4 - done")
        return perennialState, tempState, dxdt, dudt, drhodt


    
from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('integration', 'scheme', 'string', 'RK4', hint = 'The integration scheme to use. Options are: semiImplicitEuler, explicitEuler, verlet, leapfrog, RK4', required = False, export = True),
    ]