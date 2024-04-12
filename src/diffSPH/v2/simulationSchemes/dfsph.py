from diffSPH.v2.modules.pressureForce import computePressureAccelSwitch
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne

# from diffSPH.v2.modules.neighborhood import fluidNeighborSearch
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.util import countUniqueEntries
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceBarecasco
from diffSPH.v2.math import pinv2x2, scatter_sum
from diffSPH.v2.sphOps import sphOperation, sphOperationStates
from diffSPH.v2.math import scatter_sum
from diffSPH.v2.modules.densityDiffusion import renormalizedDensityGradient, computeDensityDeltaTerm
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.momentumEquation import computeMomentumEquation
from diffSPH.v2.modules.viscosity import computeViscosity
from diffSPH.v2.modules.pressureEOS import computeEOS
from diffSPH.v2.modules.pressureForce import computePressureAccel
from diffSPH.v2.modules.gravity import computeGravity
from diffSPH.v2.modules.sps import computeSPSTurbulence
from torch.profiler import record_function


from diffSPH.v2.modules.density import computeDensity
import torch
from diffSPH.v2.modules.neighborhood import neighborSearch
from diffSPH.v2.modules.normalizationMatrices import computeCovarianceMatrices
from diffSPH.v2.simulationSchemes.deltaPlus import callModule
from diffSPH.v2.modules.dfsph import dfsphSolve

def dfsphSimulationStep(state, config):
    state['fluid']['neighborhood'] = neighborSearch(state['fluid'], state['fluid'], config, priorNeighborhood = None if 'neighborhood' not in state['fluid'] else state['fluid']['neighborhood'])
    _, state['fluid']['numNeighbors'] = countUniqueEntries(state['fluid']['neighborhood']['indices'][0], state['fluid']['positions'])


    state['fluid']['densities'], _ = callModule(state, computeDensity, config, 'fluid')
    state['fluid']['velocityDiffusion'], _ = callModule(state, computeViscosity, config, 'fluid')
    state['fluid']['gravityAccel'] = computeGravity(state['fluid'], config)
    state['fluid']['momentumEquation'], _ = callModule(state, computeMomentumEquation, config, 'fluid')

    errors, pressures, state['fluid']['pressureAccel'], state['fluid']['pressures'] = dfsphSolve(state, config)
    # print(f'{state["timestep"]:3d}: {len(errors):3d} iters -> {errors[-1]:+.4e}, pressure mean: {state["fluid"]["pressures"].mean().detach().cpu().item():+.4e}')

    dudt = state['fluid']['pressureAccel'] + state['fluid']['gravityAccel'] + state['fluid']['velocityDiffusion']
    drhodt = None #state['fluid']['momentumEquation'] + state['fluid']['densityDiffusion']
    if 'boundary' not in state:
        return (state['fluid']['velocities'].clone(), dudt, drhodt), (None, None, None)
    
    boundary_dudt = state['boundary']['pressureAccel'] + state['boundary']['velocityDiffusion']
    boundary_drhodt = state['boundary']['momentumEquation']

    return (state['fluid']['velocities'].clone(), dudt, drhodt), (state['boundary']['velocities'].clone(), None, boundary_drhodt)
