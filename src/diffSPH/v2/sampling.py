from typing import Union, List
from diffSPH.v2.math import volumeToSupport
import torch

# @torch.jit.script
def evalArea(arg: Union[float, torch.Tensor], packing, dtype, device : torch.device, targetNeighbors : int, kernel, dim = 2):
    
    arg = arg if isinstance(arg, torch.Tensor) else torch.tensor(arg if isinstance(arg, float) or isinstance(arg, int) else 0.0, dtype = dtype, device = device)
    support = volumeToSupport(arg, targetNeighbors, dim)
    
    minDomain = torch.tensor([-1.1 * support] * dim, device = device, dtype = dtype)
    maxDomain = torch.tensor([1.1 * support] * dim, device = device, dtype = dtype)
    
    ii = torch.arange(maxDomain[0] / packing, device = device)
    ii = torch.hstack((ii.flip(0), ii[1:]))
    xx = (ii * packing)
    
    # print('xx', xx.shape)
    
    p = xx.view(-1,1)
    if dim == 2:    
        xx, yy = torch.meshgrid(xx,xx, indexing = 'xy')
        p = torch.stack((xx,yy), dim = -1).flatten(0,1)
    elif dim == 3:
        xx, yy, zz = torch.meshgrid(xx,xx, xx, indexing = 'xy')
        p = torch.stack((xx,yy, zz), dim = -1).flatten(0,2)
    
    rij = torch.linalg.norm(p, dim = -1)
    rij = rij[rij < support]
    k = arg * torch.sum(kernel.kernel(rij / support, torch.ones_like(rij) * support, dim))
    
    return k

    # return ((1 - rho)**2).detach().cpu().numpy()[0]

def optimizeArea(arg: Union[float, torch.Tensor], packing, dtype, device : torch.device, targetNeighbors : int, kernel, dim = 2, thresh = 1e-7, maxIter = 32):
    arg = arg if isinstance(arg, torch.Tensor) else torch.tensor(arg if isinstance(arg, float) or isinstance(arg, int) else 0.0, dtype = dtype, device = device)
    loss = 1
    for i in range(maxIter):
        arg.requires_grad = True
        eval = evalArea(arg, packing, torch.float32, 'cpu', targetNeighbors, kernel, dim)
        loss = (1-eval)**2
        loss.backward()
        arg = (arg - arg * arg.grad / targetNeighbors * 1e-3).detach()
        # print(f'iter: {i}: \tloss: {loss}, arg: {arg}')
        if loss < thresh:
            break
    # print('arg', arg)
    eval = evalArea(arg, packing, torch.float32, 'cpu', targetNeighbors, kernel, dim)
    # print('eval', eval, (1-eval)**2)
    return arg, eval, (1-eval)**2
    # print(eval)

from typing import List
def sampleRegular(
        dx: float = 2/32, dim: int = 2,
        minExtent : float | List[float] = -1, maxExtent : float | List[float] = 1, 
        
        targetNeighbors : int = 50, correctedArea : bool = False, kernel = None):
    minDomain = torch.tensor([minExtent] * dim).to(torch.float32) if isinstance(minExtent, float) or isinstance(minExtent, int) else torch.tensor(minExtent).to(torch.float32)
    maxDomain = torch.tensor([maxExtent] * dim).to(torch.float32) if isinstance(maxExtent, float) or isinstance(maxExtent, int) else torch.tensor(maxExtent).to(torch.float32)
    # dim = minDomain.shape[0]
    # dx = (maxDomain[0] - minDomain[0]) / nx          
    area = dx**(dim)
    # print(area)
    
    if correctedArea:
        area, *_ = optimizeArea(area, dx.to('cpu'), torch.float64, 'cpu', targetNeighbors, kernel, dim = dim, thresh = 1e-7**2, maxIter = 64)
    ns = [torch.ceil((maxDomain[i] - minDomain[i]) / dx).to(torch.long) for i in range(dim)]
    lins = [torch.linspace(minDomain[i] + dx / 2, maxDomain[i] - dx/2, ns[i]) for i in range(dim)]
    grid = torch.meshgrid(*lins, indexing = 'xy')
    p = torch.stack(grid, dim = -1).flatten(0,-1).view(-1,dim)

    return p, area

import numpy as np

# Function to sample particles such that their density equals a desired PDF
def samplePDF(pdf, n = 2048, numParticles = 1024, plot = False, randomSampling = False):
    xs = np.linspace(-1,1,n)
    normalized_pdf = lambda x: pdf(x) / np.sum(pdf(np.linspace(-1,1,n)))

    xs = np.linspace(-1,1,n)
    fxs = normalized_pdf(xs)
    sampled_cdf = np.cumsum(fxs) - fxs[0]
    sampled_cdf = sampled_cdf / sampled_cdf[-1] 
    inv_cdf = lambda x : np.interp(x, sampled_cdf, np.linspace(-1,1,n))

    samples = np.random.uniform(size = numParticles)
    if not randomSampling:
        samples = np.linspace(0,1,numParticles, endpoint=False)
    sampled = inv_cdf(samples)

    return torch.tensor(sampled, dtype = torch.float32).view(-1,1)




from matplotlib.path import Path
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt

from diffSPH.v2.finiteDifference import centralDifferenceStencil, continuousGradient
def filterParticlesWithSDF(p, sdf, h, threshold = 0.0):
    stencil = centralDifferenceStencil(1, 2)
    stencil = stencil.to(p.device)
    sdfValues = sdf(p.cpu()).to(p.device)
    mask = sdfValues <= threshold
    masked = p[mask]
    sdfGradient = continuousGradient(sdf, p, stencil, 0.2 * h, 1)
    sdfGradient = torch.nn.functional.normalize(sdfGradient, dim = -1)
    
    return masked, mask, sdfValues, sdfGradient
import numpy as np

def polygonToSDF(polygon, minDomain = [-1, -1], maxDomain = [1,1], sdfResolution = 128):
    scale_x = sdfResolution / (maxDomain[0] - minDomain[0])
    scale_y = sdfResolution / (maxDomain[1] - minDomain[1])
    scale = max(scale_x, scale_y)

    scaledPolygon = (polygon - np.array(minDomain)) * scale
    mask = polygon2mask((sdfResolution, sdfResolution), scaledPolygon)
    inside_distance = distance_transform_edt(mask)
    outside_distance = distance_transform_edt(1 - mask)
    signed_distance = outside_distance - inside_distance

    return signed_distance / scale

def maskWithSDF(x, sdf, minDomain, maxDomain):
    sdf_query = lambda x, sdf, minDomain, maxDomain: sdf[int((x[1] - minDomain[1]) / (maxDomain[1] - minDomain[1]) * sdf.shape[0]), int((x[0] - minDomain[0]) / (maxDomain[0] - minDomain[0]) * sdf.shape[1])]
    distances = torch.tensor([sdf_query(p, sdf, minDomain, maxDomain) for p in x.detach().cpu().numpy()])
    mask = distances < 0
    return mask, distances

def maskWithPolygon(x, polygon):
    path = Path(polygon)
    mask = torch.tensor(path.contains_points(x.detach().cpu().numpy()))
    return mask

def maskWithFunction(x, implicitFunction):
    mask = implicitFunction(x) < 0
    return mask


def sampleSDF(sdf, minExtent, maxExtent, nGrid):
    if isinstance(minExtent, torch.Tensor) and minExtent.shape[0] > 1:
        x = torch.linspace(minExtent[0], maxExtent[0], nGrid, dtype = torch.float32)
        y = torch.linspace(minExtent[1], maxExtent[1], nGrid, dtype = torch.float32)
    else:
        x = torch.linspace(minExtent, maxExtent, nGrid, dtype = torch.float32)
        y = torch.linspace(minExtent, maxExtent, nGrid, dtype = torch.float32)
    X, Y = torch.meshgrid(x, y)
    P = torch.stack([X,Y], dim=-1)
    points = P.reshape(-1,2)
    return sdf(points).reshape(nGrid, nGrid), points

from skimage import measure

def find_contour(f, minExtent, maxExtent, nGrid, level = 0):
    contours = measure.find_contours(f.numpy(), level)
    for ic in range(len(contours)):
        contours[ic][:,0] = (contours[ic][:,0]) / (f.shape[0] - 1) * (maxExtent[0] - minExtent[0]).numpy() + minExtent[0].numpy()
        contours[ic][:,1] = (contours[ic][:,1]) / (f.shape[0] - 1) * (maxExtent[1] - minExtent[1]).numpy() + minExtent[1].numpy()
    return contours

def contourLength(contour):
    return np.sum(np.linalg.norm(contour[:-1] - contour[1:], axis = -1))

def adjustSpacing(contourLength, spacing):
    numDivisions = int(contourLength / spacing)
    newSpacing = contourLength / numDivisions
    return newSpacing

def sampleContour(contour, spacing):
    cumLength = 0
    ptcls = []
    offset = spacing    
    for i in range(len(contour) -1):
        c = contour[i]
        n = contour[i+1]
        d = n - c
        curLength = np.linalg.norm(d,axis=0)
        d = d / curLength        
        curr = cumLength - offset
        while curr + spacing < cumLength + curLength - 1e-5:
            cOffset = curr + spacing - cumLength
            newP = c + d * cOffset
            ptcls.append(newP)
            curr += spacing        
        cumLength = cumLength + curLength
        offset = cumLength - curr
    return np.array(ptcls)

def sampleSDFWithParticles(sdf, minExtent, maxExtent, nGrid, dx, internalBands, externalBands):
    f, points = sampleSDF(sdf, minExtent, maxExtent, nGrid)
    sampledPointsExternal = []
    for i in range(externalBands):
        contours = find_contour(f, minExtent, maxExtent, nGrid, dx/2  + dx * (i))
        for c in contours:
            sampledPointsExternal.append(sampleContour(c, adjustSpacing(contourLength(c), dx)))
    sampledPointsExternal = np.concatenate(sampledPointsExternal, axis = 0) if len(sampledPointsExternal) > 0 else []
    sampledPointsInternal = []
    for i in range(internalBands):
        contours = find_contour(f, minExtent, maxExtent, nGrid, -dx/2 - dx * (i))
        for c in contours:
            sampledPointsInternal.append(sampleContour(c, adjustSpacing(contourLength(c), dx)))
    sampledPointsInternal = np.concatenate(sampledPointsInternal, axis = 0) if len(sampledPointsInternal) > 0 else []
    if len(sampledPointsInternal) == 0:sampledPoints = sampledPointsExternal
    elif len(sampledPointsExternal) == 0:sampledPoints = sampledPointsInternal
    else: sampledPoints = np.vstack((sampledPointsInternal, sampledPointsExternal))
    return sampledPoints, f, points, sampledPointsInternal, sampledPointsExternal


def sampleSDF_contour(sdf, minExtent, maxExtent, nGrid, dx, h, internalBands, externalBands):
    f, points = sampleSDF(sdf, minExtent, maxExtent, nGrid)
    sampledPoints = []
    sampledPointDistances = []
    sampledPointNormals = []

    for i in range(externalBands):
        contours = find_contour(f, minExtent, maxExtent, nGrid, dx/2  + dx * (i))
        for c in contours:
            s = sampleContour(c, adjustSpacing(contourLength(c), dx))
            sampledPoints.append(s)
            maskedA, maskA, sdfValues, sdfGradients = filterParticlesWithSDF(torch.tensor(s), sdf, h, -1e-4)
            sampledPointNormals.append(sdfGradients)
            sampledPointDistances.append(sdfValues)

    for i in range(internalBands):
        contours = find_contour(f, minExtent, maxExtent, nGrid, -dx/2 - dx * (i))
        for c in contours:
            s = sampleContour(c, adjustSpacing(contourLength(c), dx))
            sampledPoints.append(s)
            maskedA, maskA, sdfValues, sdfGradients = filterParticlesWithSDF(torch.tensor(s), sdf, h, -1e-4)
            sampledPointNormals.append(sdfGradients)
            sampledPointDistances.append(sdfValues)
            
    sampledPoints = np.concatenate(sampledPoints, axis = 0) if len(sampledPoints) > 0 else []
    sampledPointNormals = np.concatenate(sampledPointNormals, axis = 0) if len(sampledPointNormals) > 0 else []
    sampledPointDistances = np.concatenate(sampledPointDistances, axis = 0) if len(sampledPointDistances) > 0 else []
    volume = dx**2

    return torch.tensor(sampledPoints), torch.tensor(volume), torch.tensor(sampledPointDistances), torch.tensor(sampledPointNormals)



def emitParticlesRegular(dx, minExtent, maxExtent, config: dict):
    p, volume = sampleRegular(dx, config['domain']['dim'], minExtent.cpu().numpy(), maxExtent.cpu().numpy(), config['kernel']['targetNeighbors'], config['simulation']['correctArea'], config['kernel']['function'])

    areas = p.new_ones(p.shape[0]) * volume
    supports = p.new_ones(p.shape[0]) * volumeToSupport(volume, config['kernel']['targetNeighbors'], config['domain']['dim'])
    velocities = p.new_zeros(p.shape[0], config['domain']['dim'])
    
    return p, areas, supports, velocities

def emitParticlesSDF(sdf, dx, minExtent, maxExtent, config: dict, sdfThreshold = 0.0):
    p, volume = sampleRegular(dx, config['domain']['dim'], minExtent.cpu().numpy(), maxExtent.cpu().numpy(), config['kernel']['targetNeighbors'], config['simulation']['correctArea'], config['kernel']['function'])
    h = volumeToSupport(volume, config['kernel']['targetNeighbors'], config['domain']['dim'])

    particles, maskB, *_ = filterParticlesWithSDF(p, sdf, h, sdfThreshold)

    areas = particles.new_ones(particles.shape[0]) * volume
    supports = particles.new_ones(particles.shape[0]) * h
    velocities = particles.new_zeros(particles.shape[0], config['domain']['dim'])

    return particles, areas, supports, velocities

from diffSPH.v2.noise import generateNoise

def sampleNoise(noiseConfig):
    if 'baseFrequency' not in noiseConfig:
        noiseConfig['baseFrequency'] = 3
    if 'dim' not in noiseConfig:
        noiseConfig['dim'] = 2
    if 'octaves' not in noiseConfig:
        noiseConfig['octaves'] = 1
    if 'persistence' not in noiseConfig:
        noiseConfig['persistence'] = 0.5
    if 'lacunarity' not in noiseConfig:
        noiseConfig['lacunarity'] = 2.0
    if 'seed' not in noiseConfig:
        noiseConfig['seed'] = 234675
    if 'tileable' not in noiseConfig:
        noiseConfig['tileable'] = True
    if 'kind' not in noiseConfig:
        noiseConfig['kind'] = 'simplex'
    

    *grid, noise = generateNoise(**noiseConfig)
    return grid, noise


from diffSPH.v2.sphOps import sphOperation, sphOperationStates
from diffSPH.v2.modules.neighborhood import neighborSearch

def sampleVelocityField(noiseState, neighborhood):
    gradTerm = sphOperationStates(noiseState, noiseState, (noiseState['potential'], noiseState['potential']), 'gradient', 'difference', neighborhood=neighborhood)
    velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = -1)
    divergence = sphOperationStates(noiseState, noiseState, (noiseState['velocities'], noiseState['velocities']), 'divergence', neighborhood=neighborhood)
    return velocities, divergence

def rampDivergenceFree(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions.cpu()).to(positions.device)
#     r = sdf / d0 /2  + 0.5
    r = (sdf - offset) / d0 / 0.5 - 1
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    return (ramped /2 + 0.5) * (noise)


def rampOrthogonal(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions.cpu()).to(positions.device)
#     r = sdf / d0 /2  + 0.5
    r = (sdf - offset) / d0 
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    return (ramped) * (noise)

def filterPotentialField(sdf, noiseState, fluidConfig, kind = 'divergenceFree'):
    if kind == 'divergenceFree':
        return rampDivergenceFree(noiseState['positions'], noiseState['potential'], sdf, offset = noiseState['supports'], d0 = noiseState['supports'])
    else:
        return rampOrthogonal(noiseState['positions'], noiseState['potential'], sdf, offset = fluidConfig['particle']['dx'] / 2, d0 = noiseState['supports'])
    
from diffSPH.v2.util import countUniqueEntries
from diffSPH.v2.sdf import operatorDict

def sampleNoisyParticles(noiseConfig, config, sdfs = []):
    particlesA, volumeA = sampleRegular(config['particle']['dx'], config['domain']['dim'], config['domain']['minExtent'], config['domain']['maxExtent'], config['kernel']['targetNeighbors'], config['simulation']['correctArea'], config['kernel']['function'])
    
    noiseState = {}
    noiseState['timestep'] = 0
    noiseState['time'] = 0.
    noiseState['dt'] = config['timestep']['dt']
    area = (4 / config['particle']['nx']**2)
    area = volumeA
    grid, noiseSimplex = sampleNoise(noiseConfig)
    noiseState = {}
    noiseState ['positions'] = particlesA
    noiseState ['areas'] = particlesA.new_ones(particlesA.shape[0]) * area
    noiseState ['masses'] = noiseState['areas'] * config['fluid']['rho0']
    noiseState ['supports'] = volumeToSupport(area, config['kernel']['targetNeighbors'], config['domain']['dim']) * particlesA.new_ones(particlesA.shape[0])
    noiseState ['velocities'] = torch.zeros_like(particlesA)
    noiseState ['potential'] = noiseSimplex.flatten()
    noiseState ['index'] = torch.arange(particlesA.shape[0], device = particlesA.device)

    noiseState['numParticles'] = particlesA.shape[0]

    noiseState['neighborhood'] = neighborSearch(noiseState,noiseState, config)
    noiseState['densities'] = particlesA.new_ones(particlesA.shape[0]) * config['fluid']['rho0'] #sphOperationStates(stateA, stateB, None, 'density')
    _, noiseState['numNeighbors'] = countUniqueEntries(noiseState['neighborhood']['indices'][0], noiseState['positions'])

    for sdf in sdfs:
        noiseState['potential'] = filterPotentialField(sdf, noiseState, config, kind = 'divergenceFree')
    noiseState['velocities'], noiseState['divergence'] = sampleVelocityField(noiseState, noiseState['neighborhood'])
    mask = torch.ones_like(noiseState['fluidPotential'], dtype = torch.bool)
    noiseState['distances'] = particlesA.new_ones(particlesA.shape[0]) * np.inf
    for sdf_func in sdfs:
        _, maskA, sdfValues, _ = filterParticlesWithSDF(particlesA, sdf_func, noiseState['supports'][0], -1e-4)
        mask = mask & maskA
        noiseState['distances'] = torch.min(noiseState['distances'], sdfValues)
    # for sdf in sdfs:
        # _, maskA, _, _ = filterParticlesWithSDF(particlesA, sdf, noiseState['fluidSupports'][0], -1e-4)
        # mask = mask & maskA
    noiseState['fluidVelocities'][~mask, :] = 0
    neighborhood = neighborSearch(noiseState,noiseState, config)
    _, noiseState['numNeighbors'] = countUniqueEntries(neighborhood['indices'][0], noiseState['positions'])


    return noiseState, mask

def sampleParticles(config, sdfs = [], minExtent = None, maxExtent = None, filter = True):
    particlesA, volumeA = sampleRegular(config['particle']['dx'], config['domain']['dim'], 
                config['domain']['minExtent'] if minExtent is None else minExtent, 
                config['domain']['maxExtent'] if maxExtent is None else maxExtent, 
                config['kernel']['targetNeighbors'], config['simulation']['correctArea'], config['kernel']['function'])
    particlesA = particlesA.to(config['compute']['device'])
    volumeA = volumeA.to(config['compute']['device'])
    area = (4 / config['particle']['nx']**2)
    area = volumeA

    noiseState = {}
    noiseState['numParticles'] = particlesA.shape[0]
    noiseState['positions'] = particlesA
    noiseState['areas'] = particlesA.new_ones(particlesA.shape[0]) * area
    noiseState['pressures'] = particlesA.new_zeros(particlesA.shape[0])
    noiseState['divergence'] = particlesA.new_zeros(particlesA.shape[0])
    noiseState['masses'] = noiseState['areas'] * config['fluid']['rho0']
    noiseState['supports'] = volumeToSupport(area, config['kernel']['targetNeighbors'], config['domain']['dim']) * particlesA.new_ones(particlesA.shape[0])
    noiseState['index'] = torch.arange(particlesA.shape[0], device = particlesA.device)
    noiseState['densities'] = particlesA.new_ones(particlesA.shape[0]) * config['fluid']['rho0'] 
    noiseState['velocities'] = particlesA.new_zeros(particlesA.shape[0], config['domain']['dim'])
    noiseState['accelerations'] = particlesA.new_zeros(particlesA.shape[0], config['domain']['dim'])
    if len(sdfs) > 0:
        noiseState['distances'] = particlesA.new_ones(particlesA.shape[0]) * np.inf




    mask = torch.ones_like(noiseState['areas'], dtype = torch.bool)
    for sdf_func in sdfs:
        _, maskA, sdfValues, _ = filterParticlesWithSDF(particlesA, sdf_func, noiseState['supports'][0], -1e-4)
        mask = mask & maskA
        noiseState['distances'] = torch.min(noiseState['distances'], sdfValues)
    noiseState['velocities'][~mask, :] = 0
    if filter:
        for k in noiseState.keys():
            if isinstance(noiseState[k], torch.Tensor):
                noiseState[k] = noiseState[k][mask]
        noiseState['numParticles'] = noiseState['positions'].shape[0]
        noiseState['index'] = torch.arange(noiseState['numParticles'], device = config['compute']['device'])

    neighborhood = neighborSearch(noiseState, noiseState, config)
    _, noiseState['numNeighbors'] = countUniqueEntries(neighborhood['indices'][0], noiseState['positions'])
    noiseState['neighborhood'] = neighborhood

        

    return noiseState, mask

def sampleNoise(noiseConfig):
    if 'baseFrequency' not in noiseConfig:
        noiseConfig['baseFrequency'] = 3
    if 'dim' not in noiseConfig:
        noiseConfig['dim'] = 2
    if 'octaves' not in noiseConfig:
        noiseConfig['octaves'] = 1
    if 'persistence' not in noiseConfig:
        noiseConfig['persistence'] = 0.5
    if 'lacunarity' not in noiseConfig:
        noiseConfig['lacunarity'] = 2
    if 'seed' not in noiseConfig:
        noiseConfig['seed'] = 234675
    if 'tileable' not in noiseConfig:
        noiseConfig['tileable'] = True
    if 'kind' not in noiseConfig:
        noiseConfig['kind'] = 'simplex'
    

    *grid, noise = generateNoise(**noiseConfig)
    return grid, noise



def sampleVelocityField(noiseState, neighborhood):
    gradTerm = sphOperationStates(noiseState, noiseState, (noiseState['potential'], noiseState['potential']), operation = 'gradient', gradientMode='difference', neighborhood=neighborhood)
    velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = -1)
    divergence = sphOperationStates(noiseState, noiseState, (noiseState['velocities'], noiseState['velocities']), operation = 'divergence', neighborhood=neighborhood)
    return velocities, divergence

def rampDivergenceFree(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions)
#     r = sdf / d0 /2  + 0.5
    r = (sdf - offset) / d0 / 0.5 - 1
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    return (ramped /2 + 0.5) * (noise)


def rampOrthogonal(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions)
#     r = sdf / d0 /2  + 0.5
    r = (sdf - offset) / d0 
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    return (ramped) * (noise)

def filterPotentialField(sdf, noiseState, config, kind = 'divergenceFree'):
    if kind == 'divergenceFree':
        return rampDivergenceFree(noiseState['positions'], noiseState['potential'], sdf, offset = noiseState['supports'], d0 = noiseState['supports'])
    else:
        return rampOrthogonal(noiseState['positions'], noiseState['potential'], sdf, offset = config['particle']['dx'] / 2, d0 = noiseState['supports'])
    
import copy
from diffSPH.v2.modules.shifting import solveShifting
from diffSPH.v2.util import printState

def sampleNoisyParticles(noiseConfig, config, sdfs = [], randomizeParticles = False):
    particlesA, volumeA = sampleRegular(config['particle']['dx'], config['domain']['dim'], config['domain']['minExtent'], config['domain']['maxExtent'], config['kernel']['targetNeighbors'], config['simulation']['correctArea'], config['kernel']['function'])
    particlesA = particlesA.to(config['compute']['device'])
    volumeA = volumeA.to(config['compute']['device'])
    
    area = (4 / config['particle']['nx']**2)
    area = volumeA
    grid, noiseSimplex = sampleNoise(noiseConfig)

    noiseState = {}
    noiseState['numParticles'] = particlesA.shape[0]
    # noiseState['timestep'] = 0
    # noiseState['time'] = 0.
    # noiseState['dt'] = config['timestep']['dt']
    noiseState['positions'] = particlesA
    noiseState['areas'] = particlesA.new_ones(particlesA.shape[0]) * area
    noiseState['pressures'] = particlesA.new_zeros(particlesA.shape[0])
    noiseState['divergence'] = particlesA.new_zeros(particlesA.shape[0])
    noiseState['masses'] = noiseState['areas'] * config['fluid']['rho0']
    noiseState['supports'] = volumeToSupport(area, config['kernel']['targetNeighbors'], config['domain']['dim']) * particlesA.new_ones(particlesA.shape[0])
    noiseState['index'] = torch.arange(particlesA.shape[0], device = particlesA.device)
    noiseState['densities'] = particlesA.new_ones(particlesA.shape[0]) * config['fluid']['rho0'] 
    noiseState['velocities'] = particlesA.new_zeros(particlesA.shape[0], config['domain']['dim'])
    noiseState['accelerations'] = particlesA.new_zeros(particlesA.shape[0], config['domain']['dim'])

    if randomizeParticles:
        baseShiftingConfig = copy.deepcopy(config['shifting'])

        config['shifting']['solver'] = 'BiCGStab_wJacobi'
        # config['shifting']['solver'] = 'BiCGStab'
        config['shifting']['maxIterations'] = 64
        config['shifting']['freeSurface'] = False
        config['shifting']['summationDensity'] = False
        config['shifting']['scheme'] = 'IPS'
        config['shifting']['maxSolveIter'] = 128
        config['shifting']['initialization'] = 'zero'
        config['shifting']['threshold'] = 0.5

        positions = torch.rand(noiseState['positions'].shape, device = noiseState['positions'].device) * 2 - 1
        shiftState = {
                'positions': positions,
                'areas': noiseState['areas'],
                'densities': noiseState['densities'],\
                'numParticles': noiseState['numParticles'],
                'velocities': noiseState['velocities'],
                'masses': noiseState['masses'],
                'supports': noiseState['supports'],
            }
        dx, states = solveShifting({
            'fluid':shiftState
        }, config)
        shiftState['positions'] = positions + dx

        config['shifting'] = baseShiftingConfig

        x = shiftState['positions'].clone()

        periodic = config['domain']['periodicity']
        minDomain = config['domain']['minExtent']
        maxDomain = config['domain']['maxExtent']
        periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
        if isinstance(periodic, torch.Tensor):
            periodicity = periodic
        if isinstance(periodic, bool):
            periodicity = torch.tensor([periodic] * x.shape[1], dtype = torch.bool).to(x.device)

        mod_positions = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
        lin_x = (mod_positions[:,0] - minDomain[0]) / (maxDomain[0] - minDomain[0])
        lin_y = (mod_positions[:,1] - minDomain[1]) / (maxDomain[1] - minDomain[1])

        gridDim = 128 // 2
        linearIndex = (torch.round(lin_x * gridDim) + torch.round(lin_y * gridDim) * gridDim).to(torch.int32)
        sortedIndices = torch.argsort(linearIndex)
        sortedPositions = mod_positions[sortedIndices]
        shiftState['positions'] = sortedPositions

        noiseNeighbors = neighborSearch(shiftState, noiseState, config)

        noise = noiseSimplex.flatten().to(particlesA.device)
        noiseState['potential'] = sphOperationStates(noiseState, shiftState, (noise, noise), operation = 'interpolate', neighborhood = noiseNeighbors)
        noiseState['positions'] = shiftState['positions']



    else:
        noiseState['potential'] = noiseSimplex.flatten().to(particlesA.device)


    fluidNeighborhood = neighborSearch(noiseState, noiseState, config)

    noiseState['neighborhood'] = fluidNeighborhood
    # printState(noiseState)
    for sdf in sdfs:
        noiseState['potential'] = filterPotentialField(sdf, noiseState, config, kind = 'divergenceFrees')
    noiseState['velocities'], noiseState['divergence'] = sampleVelocityField(noiseState,noiseState['neighborhood'])
    mask = torch.ones_like(noiseState['potential'], dtype = torch.bool)
    for sdf_func in sdfs:
        _, maskA, _, _ = filterParticlesWithSDF(particlesA, operatorDict['invert'](sdf), config['particle']['support'], -1e-4)
        mask = mask & maskA
    noiseState['velocities'][~mask, :] = 0

    _, noiseState['numNeighbors'] = countUniqueEntries(fluidNeighborhood['indices'][0], noiseState['positions'])

    return noiseState, mask

# from diffSPH.v2.sampling import sampleParticles

# print(fluidState)
# print(config)
from typing import List

def sampleSDF_regular(sdf, config):
    particlesA, volumeA = sampleRegular(config['particle']['dx'], config['domain']['dim'], config['domain']['minExtent'], config['domain']['maxExtent'], config['kernel']['targetNeighbors'], config['simulation']['correctArea'], config['kernel']['function'])
    
    particlesA = particlesA.to(config['compute']['device'])
    volumeA = volumeA.to(config['compute']['device'])
    h = volumeToSupport(volumeA, config['kernel']['targetNeighbors'], config['domain']['dim'])

    # print(particlesA.shape, volumeA, h)

    maskedA, maskA, sdfValues, sdfGradients = filterParticlesWithSDF(particlesA, sdf, h, -1e-4)

    return particlesA[maskA], volumeA, sdfValues[maskA], sdfGradients[maskA]
    
# from diffSPH.v2.sampling import sampleSDF_contour
def sampleBoundaryParticles(particleState, sdfs, config, modes = ['regular']):
    
    # h = volumeToSupport(config['particle']['support'], config['kernel']['targetNeighbors'], config['domain']['dim'])
    # print(particleState)
    # print(particleState['fluidPositions'].shape)
    fluidMask = torch.ones(particleState['positions'].shape[0], dtype = torch.bool, device = config['compute']['device'])
    h = particleState['supports'].min().cpu().item()
    boundaryPositions = []
    boundaryVolumes = []
    boundaryDistances = []
    boundaryNormals = []
    boundaryBodyIDs = []
    distances = particleState['distances'] if 'distances' in particleState else torch.ones(particleState['positions'].shape[0], dtype = torch.float32, device = config['compute']['device']) * 1e8
    normals = particleState['normals'] if 'normals' in particleState else torch.zeros(particleState['positions'].shape[0], config['domain']['dim'], dtype = torch.float32, device = config['compute']['device'])

    for i, (sdf, mode) in enumerate(zip(sdfs, modes)):
        maskedA, maskA, sdfValues, sdfGradients = filterParticlesWithSDF(particleState['positions'], operatorDict['invert'](sdf), config['particle']['support'], -1e-4)
        fluidMask = fluidMask & maskA

        normals[sdfValues < distances,:] = sdfGradients[sdfValues < distances,:]
        distances[sdfValues < distances] = sdfValues[sdfValues < distances] 
        if mode == 'regular':
            boundaryPosition, boundaryVolume, boundaryDistance, boundaryNormal = sampleSDF_regular(sdf, config)
        else:
            boundaryPosition, boundaryVolume, boundaryDistance, boundaryNormal = sampleSDF_contour(sdf, config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), 255, h =h ,dx = config['particle']['dx'], internalBands= 3, externalBands=0)
        boundaryPosition = boundaryPosition.to(config['compute']['device']).to(config['compute']['dtype'])
        boundaryPositions.append(boundaryPosition)
        boundaryVolumes.append(boundaryVolume.to(config['compute']['dtype']) * boundaryPosition.new_ones(boundaryPosition.shape[0]))
        boundaryDistances.append(boundaryDistance.to(config['compute']['device']).to(config['compute']['dtype']))
        boundaryNormals.append(boundaryNormal.to(config['compute']['device']).to(config['compute']['dtype']))
        boundaryBodyIDs.append(boundaryPosition.new_ones(boundaryPosition.shape[0]).to(config['compute']['device']) * i)

    boundaryPositions = torch.cat(boundaryPositions, dim = 0)
    boundaryVolumes = torch.cat(boundaryVolumes, dim = 0)
    boundaryDistances = torch.cat(boundaryDistances, dim = 0)
    boundaryNormals = torch.cat(boundaryNormals, dim = 0)
    boundaryBodyIDs = torch.cat(boundaryBodyIDs, dim = 0)

    particleState['distances'] = distances
    particleState['normals'] = normals

    return boundaryPositions, boundaryVolumes, boundaryDistances, boundaryNormals, boundaryBodyIDs, fluidMask

# boundaryParticles, boundaryVolumes, boundaryDistances, boundaryNormals, boundaryBodyIDs, fluidMask = sampleBoundaryParticles(fluidState, [sdf], config, 'sdf')

# for k in fluidState:
#     # print(k)
#     if isinstance(fluidState[k], torch.Tensor):
#         fluidState[k] = fluidState[k][fluidMask]
# fluidState['numParticles'] = fluidState['positions'].shape[0]

def processBoundarySDFs(fluidState, config, sdfs, samplings = None):
    if samplings is None:   
        samplings = ['regular' for _ in sdfs]
    if not isinstance(samplings, List):
        samplings = [samplings for _ in sdfs]
        
    boundaryParticles, boundaryVolumes, boundaryDistances, boundaryNormals, boundaryBodyIDs, fluidMask = sampleBoundaryParticles(fluidState, sdfs, config, samplings)
    for k in fluidState:
        if isinstance(fluidState[k], torch.Tensor):
            fluidState[k] = fluidState[k][fluidMask]
    fluidState['numParticles'] = fluidState['positions'].shape[0]

    boundaryState = {
        'positions': boundaryParticles,
        'areas': boundaryVolumes,
        'pressures': boundaryVolumes.new_zeros(boundaryVolumes.shape[0]),
        'masses': boundaryVolumes * config['fluid']['rho0'],
        'supports': boundaryVolumes.new_ones(boundaryVolumes.shape[0]) * config['particle']['support'],
        'index': torch.arange(boundaryParticles.shape[0], device = config['compute']['device']),

        'densities': boundaryVolumes.new_ones(boundaryVolumes.shape[0]) * config['fluid']['rho0'],
        'velocities': boundaryVolumes.new_zeros(boundaryVolumes.shape[0], config['domain']['dim']),
        'accelerations': boundaryVolumes.new_zeros(boundaryVolumes.shape[0], config['domain']['dim']),

        'numParticles': boundaryParticles.shape[0],

        'distances': boundaryDistances,
        'normals': boundaryNormals,
        'bodyIDs': boundaryBodyIDs,
    }

    boundaryState['neighborhood'] = neighborSearch(boundaryState, boundaryState, config)
    _, boundaryState['numNeighbors'] = countUniqueEntries(boundaryState['neighborhood']['indices'][0], boundaryState['positions'])

    return boundaryState