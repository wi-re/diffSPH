
from diffSPH.v2.modules.pressureForce import computePressureAccelSwitch
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne

from diffSPH.v2.modules.neighborhood import fluidNeighborSearch
from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.util import countUniqueEntries
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceBarecasco
from diffSPH.v2.math import pinv2x2, scatter_sum
from diffSPH.v2.sphOps import sphOperation, sphOperationFluidState
from diffSPH.v2.math import scatter_sum
from diffSPH.v2.modules.densityDiffusion import renormalizedDensityGradient, computeDensityDeltaTerm
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.momentumEquation import computeMomentumEquation
from diffSPH.v2.modules.viscosity import computeViscosity
from diffSPH.v2.modules.pressureEOS import computeEOS
from diffSPH.v2.modules.pressureForce import computePressureAccel
from diffSPH.v2.modules.gravity import computeGravity

def simulationStep(simulationState, config):
    simulationState['fluidNeighborhood'] = fluidNeighborSearch(simulationState, config)
    _, simulationState['fluidNumNeighbors'] = countUniqueEntries(simulationState['fluidNeighborhood']['indices'][0], simulationState['fluidPositions'])
    
    simulationState['fluidL'], _, simulationState['L.EVs'] = computeNormalizationMatrices(simulationState, config)
    simulationState['fluidGradRho^L'] = renormalizedDensityGradient(simulationState, config)
    # simulationState['fluidGradRho'] = sphOperationFluidState(simulationState, (simulationState['fluidDensities'], simulationState['fluidDensities']), operation = 'gradient', gradientMode='difference')
    
    simulationState['fluidDensityDiffusion'] = computeDensityDeltaTerm(simulationState, config)
    simulationState['fluidMomentumEquation'] = computeMomentumEquation(simulationState, config)

    simulationState['fluidVelocityDiffusion'] = computeViscosity(simulationState, config)
    simulationState['fluidPressures'] = computeEOS(simulationState, config)

    if config['deltaSPH']['pressureSwitch']:
        # simulationState['fluidNormals'], simulationState['fluidLambdas'] = computeNormalsMaronne(simulationState, config)
        simulationState['fluidFreeSurface']= detectFreeSurfaceBarecasco(simulationState, config)
        (i,j) = simulationState['fluidNeighborhood']['indices']
        numParticles = simulationState['numParticles']
        simulationState['fluidSurfaceMask'] = scatter_sum(simulationState['fluidFreeSurface'][j], i, dim = 0, dim_size = numParticles)
        simulationState['fluidPressureAccel'] = computePressureAccelSwitch(simulationState, config)
    else:
        simulationState['fluidPressureAccel'] = computePressureAccel(simulationState, config)

    simulationState['fluidDivergence'] = sphOperationFluidState(simulationState, (simulationState['fluidVelocities'], simulationState['fluidVelocities']), 'divergence')

    simulationState['fluidGravityAccel'] = computeGravity(simulationState, config)

    dudt = simulationState['fluidPressureAccel'] + simulationState['fluidGravityAccel'] + simulationState['fluidVelocityDiffusion']
    drhodt = simulationState['fluidMomentumEquation'] + simulationState['fluidDensityDiffusion']
    
    return simulationState['fluidVelocities'].clone(), dudt, drhodt

from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('deltaSPH', 'pressureSwitch', bool, False, required = False, export = True, hint = 'Switches the pressure force calculation to the Antuono Correction'),
    ]