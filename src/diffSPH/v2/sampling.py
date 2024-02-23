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
        area, *_ = optimizeArea(area, dx, torch.float64, 'cpu', targetNeighbors, kernel, dim = dim, thresh = 1e-7**2, maxIter = 64)
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
    sdfValues = sdf(p)
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
    sampledPointsExternal = np.concatenate(sampledPointsExternal, axis = 0)
    sampledPointsInternal = []
    for i in range(internalBands):
        contours = find_contour(f, minExtent, maxExtent, nGrid, -dx/2 - dx * (i))
        for c in contours:
            sampledPointsInternal.append(sampleContour(c, adjustSpacing(contourLength(c), dx)))
    sampledPointsInternal = np.concatenate(sampledPointsInternal, axis = 0)
    sampledPoints = np.vstack((sampledPointsInternal, sampledPointsExternal))
    return sampledPoints, f, points, sampledPointsInternal, sampledPointsExternal


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