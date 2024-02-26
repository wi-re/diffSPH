import torch
from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum

# See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
def detectFreeSurfaceMaronne(fluidState, simConfig):
    particles = fluidState['fluidPositions']
    n = fluidState['fluidNormals'] if 'fluidNormals' in fluidState else computeNormalsMaronne(fluidState, simConfig)[0]
    numParticles = particles.shape[0]
    supports = fluidState['fluidSupports']
    periodicity = simConfig['domain']['periodicity']
    domainMin = simConfig['domain']['minExtent']
    domainMax = simConfig['domain']['maxExtent']

    i,j = fluidState['fluidNeighborhood']['indices']
    rij = fluidState['fluidNeighborhood']['distances']
    hij = fluidState['fluidNeighborhood']['supports']

    
    T = particles + n * supports.view(-1,1) / 3

    tau = torch.vstack((-n[:,1], n[:,0])).mT
    xjt = particles[j] - T[i]
    xjt = torch.stack([xjt[:,i] if not periodicity[i] else mod(xjt[:,i], domainMin[i], domainMax[i]) for i in range(xjt.shape[1])], dim = -1)

    condA1 = rij >= np.sqrt(2) * hij / 3
    condA2 = torch.linalg.norm(xjt, dim = -1) <= hij / 3
    condA = (condA1 & condA2) & (i != j)
    cA = scatter_sum(condA, i, dim = 0, dim_size = numParticles)

    condB1 = rij < np.sqrt(2) * hij / 3
    condB2 = torch.abs(torch.einsum('ij,ij->i', -n[i], xjt)) + torch.abs(torch.einsum('ij,ij->i', tau[i], xjt)) < hij / 3
    condB = (condB1 & condB2) & (i != j)
    cB = scatter_sum(condB, i, dim = 0, dim_size = numParticles)
    
    fs = torch.where(~cA & ~cB, 1.,0.)
    return fs, cA, cB

# See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
def computeNormalsMaronne(fluidState, simConfig):    
    ones = fluidState['fluidPositions'].new_ones(fluidState['fluidPositions'].shape[0])
    term = sphOperationFluidState(fluidState, (ones, ones), operation = 'gradient', gradientMode='naive')
    L, lambdas = (fluidState['fluidL'], fluidState['L.EVs']) if 'fluidL' in fluidState else computeNormalizationMatrices(fluidState, simConfig)

    nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
    n = -torch.nn.functional.normalize(nu, dim = -1)
    lMin = torch.min(torch.abs(lambdas), dim = -1).values

    return n, lMin

def expandFreeSurfaceMask(fluidState, simConfig):
    fs = fluidState['fluidFreeSurface']
    i,j = fluidState['fluidNeighborhood']['indices']
    numParticles = fluidState['fluidPositions'].shape[0]

    fsm = scatter_sum(fs[j], i, dim = 0, dim_size = numParticles)
    return fsm > 0

def computeColorField(fluidState, simConfig):
    ones = fluidState['fluidPositions'].new_ones(fluidState['fluidPositions'].shape[0])
    color = sphOperationFluidState(fluidState, (ones, ones), operation = 'interpolate')
    return color
def computeColorFieldGradient(fluidState, simConfig):
    color = sphOperationFluidState(fluidState, (fluidState['fluidColor'], fluidState['fluidColor']), operation = 'gradient', gradientMode = 'difference')
    return color

def detectFreeSurfaceColorFieldGradient(fluidState, simConfig):
    gradColorField = fluidState['fluidColorGradient']
    fs = torch.linalg.norm(gradColorField, dim = -1) > simConfig['surfaceDetection']['colorFieldGradientThreshold'] * fluidState['fluidSupports']
    return fs

def detectFreeSurfaceColorField(fluidState, simConfig):
    colorField = fluidState['fluidColor']
    numParticles = fluidState['numParticles']
    i, j = fluidState['fluidNeighborhood']['indices']
    nj = scatter_sum(torch.ones_like(j), i, dim = 0, dim_size = numParticles)
    colorFieldMean = scatter_sum(colorField[j], i, dim = 0, dim_size = numParticles) / nj
    fs = torch.where((colorField < colorFieldMean) & (nj < simConfig['kernel']['targetNeighbors'] * simConfig['surfaceDetection']['colorFieldThreshold']), 1., 0.)
    return fs

# Barecasco et al 2013: Simple free-surface detection in two and three-dimensional SPH solver
def detectFreeSurfaceBarecasco(simulationState, simConfig):
    xij = simulationState['fluidNeighborhood']['vectors']
    (i,j) = simulationState['fluidNeighborhood']['indices']
    numParticles = simulationState['numParticles']

    coverVector = scatter_sum(-xij, i, dim = 0, dim_size = numParticles)
    normalized = torch.nn.functional.normalize(coverVector)
    angle = torch.arccos(torch.einsum('ij,ij->i', xij, normalized[i]))
    threshold = simConfig['surfaceDetection']['BarecascoThreshold']
    condition = (angle <= threshold / 2) & (i != j) | (torch.linalg.norm(normalized, dim = -1)[i] <= 1e-5)
    fs = ~scatter_sum(condition, i, dim = 0, dim_size = numParticles)
    return fs

from torch_scatter import scatter
def computeSurfaceDistance(simulationState, simConfig):
    surfaceDistance = simulationState['freeSurface'].new_zeros(simulationState['numParticles'], dtype = simConfig['compute']['dtype'])
    surfaceDistance[:] = 1e4
    surfaceDistance[simulationState['freeSurface']] = simConfig['particle']['dx']

    (i,j) = simulationState['fluidNeighborhood']['indices']

    for step in range(simConfig['surfaceDetection']['distanceIterations']):
        distance = surfaceDistance[j] + simulationState['fluidNeighborhood']['distances'] * simulationState['fluidNeighborhood']['supports']
        newDistance = scatter(distance, i, dim = 0, reduce = 'min', dim_size = simulationState['numParticles'])
        update = torch.mean((newDistance - surfaceDistance)**2)
        print(update)
        if torch.all(torch.abs(newDistance - surfaceDistance) < simConfig['particle']['defaultSupport'] / 4):
            break
        surfaceDistance = newDistance

    return surfaceDistance

def getStableSurfaceNormal(simulationState, simConfig):
    return sphOperationFluidState(simulationState, (simulationState['surfaceDistance'], simulationState['surfaceDistance']), operation = 'gradient', gradientMode = 'symmetric')
    

from diffSPH.parameter import Parameter
def getModuleFunctions():
    return {
        'computeNormalsMaronne': computeNormalsMaronne,
        'detectFreeSurfaceMaronne': detectFreeSurfaceMaronne,
        'expandFreeSurfaceMask': expandFreeSurfaceMask,
        'computeColorField': computeColorField,
        'computeColorFieldGradient': computeColorFieldGradient,
        'detectFreeSurfaceColorFieldGradient': detectFreeSurfaceColorFieldGradient,
        'detectFreeSurfaceColorField': detectFreeSurfaceColorField,
        'detectFreeSurfaceBarecasco': detectFreeSurfaceBarecasco

    }
def getParameters():
    return [
        Parameter('surfaceDetection', 'colorFieldGradientThreshold', float, 10.0, required = False,export = False, hint = 'Threshold for the color field gradient to detect free surface'),
        Parameter('surfaceDetection', 'colorFieldThreshold', float, 0.8, required = False,export = False, hint = 'Threshold for the free surface detection using mean neighborhood sizes'),
        Parameter('surfaceDetection', 'BarecascoThreshold', float, np.pi / 3, required = False, export = False, hint = 'Threshold for the free surface detection using Barecasco et al. method'),
        Parameter('surfaceDetection', 'distanceIterations', int, 16, required = False, export = False, hint = 'Number of iterations to compute the surface distance')
    ]