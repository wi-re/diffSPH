import torch
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum
from torch.profiler import record_function

# See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
def detectFreeSurfaceMaronne(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Free Surface Detection (Maronne)"):
        particles = stateA['positions']
        n = stateA['normals'] if 'normals' in stateA else computeNormalsMaronne(stateA, stateB, neighborhood, simConfig)[0]
        numParticles = particles.shape[0]
        supports = stateA['supports']
        periodicity = simConfig['domain']['periodicity']
        domainMin = simConfig['domain']['minExtent']
        domainMax = simConfig['domain']['maxExtent']

        i,j = neighborhood['indices']
        rij = neighborhood['distances']
        hij = neighborhood['supports']

        
        T = particles + n * supports.view(-1,1) / simConfig['kernel']['kernelScale'] / 3

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
        
        fs = torch.where(~cA & ~cB & (torch.linalg.norm(n, dim = -1) > 0.5), 1.,0.)
        return fs, cA, cB

# See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
def computeNormalsMaronne(stateA, stateB, neighborhood, simConfig):    
    with record_function("[SPH] - Normal Computation (Maronne)"):
        ones = stateA['positions'].new_ones(stateA['positions'].shape[0])
        term = sphOperationStates(stateA, stateB, (ones, ones), operation = 'gradient', gradientMode='naive', neighborhood = neighborhood)
        L, lambdas = (stateA['L'], stateA['L.EVs']) if 'L' in stateA else computeNormalizationMatrices(stateA, stateB, neighborhood, simConfig)

        nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
        n = -torch.nn.functional.normalize(nu, dim = -1)
        lMin = torch.min(torch.abs(lambdas), dim = -1).values

        return n, lMin

def expandFreeSurfaceMask(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Free Surface Mask Expansion"):
        fs = stateA['freeSurface']
        i,j = neighborhood['indices']
        numParticles = stateA['positions'].shape[0]

        fsm = torch.clone(fs)
        for ii in range(simConfig['surfaceDetection']['expansionIterations']):
            fsm = scatter_sum(fsm[j], i, dim = 0, dim_size = numParticles)
        return fsm > 0

def computeColorField(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Color Field Computation"):
        ones = stateA['positions'].new_ones(stateA['positions'].shape[0])
        color = sphOperationStates(stateA, stateB, (ones, ones), operation = 'interpolate', neighborhood = neighborhood)
        return color
def computeColorFieldGradient(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Color Field Gradient Computation"):
        color = sphOperationStates(stateA, stateB, (stateA['color'], stateB['color']), operation = 'gradient', gradientMode = 'difference', neighborhood = neighborhood)
        return color

def detectFreeSurfaceColorFieldGradient(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Free Surface Detection (Color Field Gradient)"):
        gradColorField = stateA['colorGradient']
        fs = torch.linalg.norm(gradColorField, dim = -1) > simConfig['surfaceDetection']['colorFieldGradientThreshold'] * stateA['supports'] / simConfig['kernel']['kernelScale']
        return fs

def detectFreeSurfaceColorField(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Free Surface Detection (Color Field)"):
        colorField = stateA['color']
        numParticles = stateA['numParticles']
        i, j = neighborhood['indices']
        nj = scatter_sum(torch.ones_like(j), i, dim = 0, dim_size = numParticles)
        colorFieldMean = scatter_sum(colorField[j], i, dim = 0, dim_size = numParticles) / nj
        fs = torch.where((colorField < colorFieldMean) & (nj < simConfig['kernel']['targetNeighbors'] * simConfig['surfaceDetection']['colorFieldThreshold']), 1., 0.)
        return fs

# Barecasco et al 2013: Simple free-surface detection in two and three-dimensional SPH solver
def detectFreeSurfaceBarecasco(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Free Surface Detection (Barecasco)"):
        xij = neighborhood['vectors']
        (i,j) = neighborhood['indices']
        numParticles = stateA['numParticles']

        coverVector = scatter_sum(-xij, i, dim = 0, dim_size = numParticles)
        normalized = torch.nn.functional.normalize(coverVector)
        angle = torch.arccos(torch.einsum('ij,ij->i', xij, normalized[i]))
        threshold = simConfig['surfaceDetection']['BarecascoThreshold']
        condition = (angle <= threshold / 2) & (i != j) | (torch.linalg.norm(normalized, dim = -1)[i] <= 1e-5)
        fs = ~scatter_sum(condition, i, dim = 0, dim_size = numParticles)
        return fs

# from torch_scatter import scatter
def computeSurfaceDistance(stateA, stateB, neighborhood, simConfig):
    with record_function("[SPH] - Surface Distance Computation"):
        surfaceDistance = stateA['freeSurface'].new_zeros(stateA['numParticles'], dtype = simConfig['compute']['dtype'])
        surfaceDistance[:] = 1e4
        surfaceDistance[stateA['freeSurface']] = simConfig['particle']['dx']

        (i,j) = neighborhood['indices']

        for step in range(simConfig['surfaceDetection']['distanceIterations']):
            distance = surfaceDistance[j] + neighborhood['distances'] * neighborhood['supports']
            newDistance = distance.new_zeros(stateA['numParticles'], dtype = simConfig['compute']['dtype'])
            newDistance.index_reduce_(dim = 0, index = i, source = distance, include_self = False, reduce = 'amin')
            # newDistance = scatter(distance, i, dim = 0, reduce = 'min', dim_size = simulationState['numParticles'])
            update = torch.mean((newDistance - surfaceDistance)**2)
            print(update)
            if torch.all(torch.abs(newDistance - surfaceDistance) < simConfig['particle']['defaultSupport'] / 4):
                break
            surfaceDistance = newDistance

        return surfaceDistance

def getStableSurfaceNormal(stateA, stateB, neighborhood, simConfig):
    return sphOperationStates(stateA, stateB, (stateA['surfaceDistance'], stateB['surfaceDistance']), operation = 'gradient', gradientMode = 'symmetric', neighborhood = neighborhood)
    

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
        Parameter('surfaceDetection', 'distanceIterations', int, 16, required = False, export = False, hint = 'Number of iterations to compute the surface distance'),
        Parameter('surfaceDetection', 'expansionIterations', int, 2, required = False, export = False, hint = 'Number of iterations to expand the free surface mask')
    ]