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
        if loss < thresh:
            break
    # print('arg', arg)
    eval = evalArea(arg, packing, torch.float32, 'cpu', targetNeighbors, kernel, dim)
    return arg, eval, (1-eval)**2
    # print(eval)

from typing import List
def sampleRegular(nx: int = 32, minExtent : float | List[float] = -1, maxExtent : float | List[float] = 1, targetNeighbors : int = 50, correctedArea : bool = False, kernel = None):
    minDomain = torch.tensor([minExtent]).to(torch.float32) if isinstance(minExtent, float) or isinstance(minExtent, int) else torch.tensor(minExtent).to(torch.float32)
    maxDomain = torch.tensor([maxExtent]).to(torch.float32) if isinstance(maxExtent, float) or isinstance(maxExtent, int) else torch.tensor(maxExtent).to(torch.float32)
    dim = minDomain.shape[0]
    dx = (maxDomain[0] - minDomain[0]) / nx          
    area = dx**(dim)
    # print(area)
    
    if correctedArea:
        area, *_ = optimizeArea(area, dx, torch.float64, 'cpu', targetNeighbors, kernel, dim = dim, thresh = 1e-7, maxIter = 32)
    ns = [torch.ceil((maxDomain[i] - minDomain[i]) / dx).to(torch.long) for i in range(minDomain.shape[0])]
    lins = [torch.linspace(minDomain[i] + dx / 2, maxDomain[i] - dx/2, ns[i]) for i in range(minDomain.shape[0])]

    if dim == 1:
        return lins[0].view(-1,1)
    elif dim == 2:    
        xx, yy = torch.meshgrid(lins[0],lins[1], indexing = 'xy')
        p = torch.stack((xx,yy), dim = -1).flatten(0,1)
        return p
    elif dim == 3:
        xx, yy, zz = torch.meshgrid(lins[0],lins[1], lins[2], indexing = 'xy')
        p = torch.stack((xx,yy, zz), dim = -1).flatten(0,2)
        return p
    

from matplotlib.path import Path
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt

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