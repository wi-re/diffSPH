
# from diffSPH.v2.modules.pressureForce import computePressureAccelSwitch
# from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne

# from diffSPH.v2.modules.neighborhood import fluidNeighborSearch
# from diffSPH.v2.sphOps import sphOperationStates
# from diffSPH.v2.util import countUniqueEntries
# from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
# from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceBarecasco
# from diffSPH.v2.math import pinv2x2, scatter_sum
# from diffSPH.v2.sphOps import sphOperation, sphOperationStates
# from diffSPH.v2.math import scatter_sum
# from diffSPH.v2.modules.densityDiffusion import renormalizedDensityGradient, computeDensityDeltaTerm
# from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
# from diffSPH.v2.modules.momentumEquation import computeMomentumEquation
# from diffSPH.v2.modules.viscosity import computeViscosity
# from diffSPH.v2.modules.pressureEOS import computeEOS
# from diffSPH.v2.modules.pressureForce import computePressureAccel
# from diffSPH.v2.modules.gravity import computeGravity
# from diffSPH.v2.modules.sps import computeSPSTurbulence
# from torch.profiler import record_function

# def simulationStep(simulationState, config):
#     with record_function("[SPH] - deltaSPH"):
#         with record_function("[SPH] - deltaSPH (1 - Neighbor Search)"):
#             simulationState['fluidNeighborhood'] = fluidNeighborSearch(simulationState, config)
#             _, simulationState['fluidNumNeighbors'] = countUniqueEntries(simulationState['fluidNeighborhood']['indices'][0], simulationState['fluidPositions'])
        
#         with record_function("[SPH] - deltaSPH (2 - Normalization Matrices)"):
#             simulationState['fluidL'], _, simulationState['L.EVs'] = computeNormalizationMatrices(simulationState, config)
#             simulationState['fluidGradRho^L'] = renormalizedDensityGradient(simulationState, config)
#         # simulationState['fluidGradRho'] = sphOperationStates(stateA, stateB, (simulationState['fluidDensities'], simulationState['fluidDensities']), operation = 'gradient', gradientMode='difference')
        
#         with record_function("[SPH] - deltaSPH (3 - Diffusion)"):
#             simulationState['fluidDensityDiffusion'] = computeDensityDeltaTerm(simulationState, config)
#             simulationState['fluidVelocityDiffusion'] = computeViscosity(simulationState, config)

#         with record_function("[SPH] - deltaSPH (4 - Momentum + Pressure)"):
#             simulationState['fluidMomentumEquation'] = computeMomentumEquation(simulationState, config)
#             simulationState['fluidPressures'] = computeEOS(simulationState, config)

#             if config['deltaSPH']['pressureSwitch']:
#                 # simulationState['fluidNormals'], simulationState['fluidLambdas'] = computeNormalsMaronne(simulationState, config)
#                 simulationState['fluidFreeSurface']= detectFreeSurfaceBarecasco(simulationState, config)
#                 (i,j) = simulationState['fluidNeighborhood']['indices']
#                 numParticles = simulationState['numParticles']
#                 simulationState['fluidSurfaceMask'] = scatter_sum(simulationState['fluidFreeSurface'][j], i, dim = 0, dim_size = numParticles)
#                 simulationState['fluidPressureAccel'] = computePressureAccelSwitch(simulationState, config)
#             else:
#                 simulationState['fluidPressureAccel'] = computePressureAccel(simulationState, config)

#         with record_function("[SPH] - deltaSPH (5 - Compute Divergence)"):
#             simulationState['fluidDivergence'] = sphOperationStates(stateA, stateB, (simulationState['fluidVelocities'], simulationState['fluidVelocities']), 'divergence')
#         with record_function("[SPH] - deltaSPH (6 - Gravity)"):
#             simulationState['fluidGravityAccel'] = computeGravity(simulationState, config)
#         if config['SPS']['active']:
#             with record_function("[SPH] - deltaSPH (7 - SPS Turbulence)"):
#                 simulationState['fluidSPSTurbulence'] = computeSPSTurbulence(simulationState, config)
#                 simulationState['fluidVelocityDiffusion'] += simulationState['fluidSPSTurbulence']

#         with record_function("[SPH] - deltaSPH (8 - Update)"):
#             dudt = simulationState['fluidPressureAccel'] + simulationState['fluidGravityAccel'] + simulationState['fluidVelocityDiffusion']
#             drhodt = simulationState['fluidMomentumEquation'] + simulationState['fluidDensityDiffusion']
            
#             return simulationState['fluidVelocities'].clone(), dudt, drhodt

# from diffSPH.parameter import Parameter
# def getParameters():
#     return [
#         Parameter('deltaSPH', 'pressureSwitch', bool, False, required = False, export = True, hint = 'Switches the pressure force calculation to the Antuono Correction'),
#     ]