from diffSPH.v2.finiteDifference import computeGradient
from diffSPH.v2.sdf import getSDF, sdfFunctions, operatorDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch


def plotSDF(fx, X, Y, extent, dim, ngrid = 255):
    fig, axis = plt.subplots(1, 3, figsize=(14,4), sharex = False, sharey = False, squeeze = False)

    fx = fx.reshape(ngrid, ngrid)
    output = computeGradient(fx, extent, dim, 1)
    # Plot the isocontours of fx
    axis[0,0].contour(X.numpy(), Y.numpy(), fx.numpy(), levels=[0], colors='black')


    # sdfValues = sdCircle(P, 1.0)
    # sdfGradient = gradient(sdfValues, 4, 1, 2)
    # sdfGradient = torch.stack([centralDifference(sdfValues, 1/ngrid, 1, 4), centralDifference(sdfValues, 1/ngrid, 1, 4)], axis=-1)
    # sdfValues = fx
    # sdfGradient = output
    # print(sdfGradient.shape)

    im = axis[0,0].pcolormesh(X.numpy(), Y.numpy(), fx.numpy(), cmap='Spectral',vmin = - torch.max(torch.abs(fx)).numpy(), vmax = torch.max(torch.abs(fx)).numpy())
    axis[0,0].set_title("SDF")
    axis[0,0].set_aspect('equal', 'box')
    divider = make_axes_locatable(axis[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    spacing = 16
    axis[0,0].quiver(X[::spacing,::spacing].numpy(), Y[::spacing,::spacing].numpy(), output[::spacing,::spacing,0].numpy(), output[::spacing,::spacing,1].numpy())

    im = axis[0,1].pcolormesh(X.numpy(), Y.numpy(), output[:,:,0].numpy(), cmap='viridis')
    axis[0,1].set_title("SDF Gradient X")
    axis[0,1].set_aspect('equal', 'box')
    divider = make_axes_locatable(axis[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axis[0,2].pcolormesh(X.numpy(), Y.numpy(), output[:,:,1].numpy(), cmap='viridis')
    axis[0,2].set_title("SDF Gradient Y")
    axis[0,2].set_aspect('equal', 'box')
    divider = make_axes_locatable(axis[0,2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # im = axis[0,3].imshow(torch.linalg.norm(output[:,:], dim = -1), extent=(-2, 2, -2, 2), origin='lower', cmap='inferno')
    # axis[0,3].set_title("SDF Gradient Magnitude")
    # divider = make_axes_locatable(axis[0,3])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # axis[0,3].set_title("SDF Gradient")
    # axis[0,3].set_aspect('equal', 'box')
    # axis[0,3].set_xlim(-2, 2)
    # axis[0,3].set_ylim(-2, 2)

    fig.tight_layout()

def scatterPlot(fig, axis, p, c, domainMin, domainMax, label = None, periodic = True, cmap = 'viridis', s = None):
    s = 5000 / p.shape[0] if s is None else s
    dim = p.shape[1]
    pos_x = torch.stack([p[:,i] if not periodic else torch.remainder(p[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i in range(dim)], dim = 1)
    # pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)

    sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = c.detach().cpu().numpy(), cmap = cmap)
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    cb = fig.colorbar(sc, cax=cax1,orientation='vertical')
    if label is not None:
        axis.set_title(label)
    if p.shape[1] > 1:
        square = patches.Rectangle((domainMin[0].detach().cpu().numpy(), domainMin[1].detach().cpu().numpy()), domainMax[0].detach().cpu().numpy() - domainMin[0].detach().cpu().numpy(), domainMax[1].detach().cpu().numpy() - domainMin[1].detach().cpu().numpy(), linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')
    axis.set_xlim(-1.05,1.05)
    axis.set_ylim(-1.05,1.05)
    return sc, cb

def scatterPlotSymmetric(fig, axis, p, c, domainMin, domainMax, label = None, periodic = True, cmap = 'coolwarm', s = None):
    s = 5000 / p.shape[0] if s is None else s
    dim = p.shape[1]
    pos_x = torch.stack([p[:,i] if not periodic else torch.remainder(p[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i in range(dim)], dim = 1)
    # pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)

    sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = c.detach().cpu().numpy(), vmin = -c.abs().detach().cpu().numpy().max(), vmax = c.abs().detach().cpu().numpy().max(), cmap = cmap)
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    cb = fig.colorbar(sc, cax=cax1,orientation='vertical')
    if label is not None:
        axis.set_title(label)
    if p.shape[1] > 1:
        square = patches.Rectangle((domainMin[0].detach().cpu().numpy(), domainMin[1].detach().cpu().numpy()), domainMax[0].detach().cpu().numpy() - domainMin[0].detach().cpu().numpy(), domainMax[1].detach().cpu().numpy() - domainMin[1].detach().cpu().numpy(), linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')
    axis.set_xlim(-1.05,1.05)
    axis.set_ylim(-1.05,1.05)
    return sc, cb


def scatterPlotFluid(fig, axis, state, config, q, label = None, cmap = 'viridis', s = None):
    x = state['fluidPositions'].detach().cpu()
    s = 5000 / x.shape[0] if s is None else s
    dim = x.shape[1]
    domainMin = config['domain']['minExtent']
    domainMax = config['domain']['maxExtent']
    periodicity = config['domain']['periodicity']
    pos_x = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicity)], dim = 1)# pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)

    sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = q.detach().cpu().numpy(), cmap = cmap)
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    cb = fig.colorbar(sc, cax=cax1,orientation='vertical')
    if label is not None:
        axis.set_title(label)
    if x.shape[1] > 1:
        square = patches.Rectangle((domainMin[0].detach().cpu().numpy(), domainMin[1].detach().cpu().numpy()), domainMax[0].detach().cpu().numpy() - domainMin[0].detach().cpu().numpy(), domainMax[1].detach().cpu().numpy() - domainMin[1].detach().cpu().numpy(), linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')
    axis.set_xlim(-1.05,1.05)
    axis.set_ylim(-1.05,1.05)
    return sc, cb

def scatterPlotFluidSymmetric(fig, axis, state, config, q, label = None, cmap = 'viridis', s = None):
    x = state['fluidPositions'].detach().cpu()
    s = 5000 / x.shape[0] if s is None else s
    dim = x.shape[1]
    domainMin = config['domain']['minExtent']
    domainMax = config['domain']['maxExtent']
    periodicity = config['domain']['periodicity']
    pos_x = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicity)], dim = 1)# pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)

    sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = q.detach().cpu().numpy(), cmap = cmap, vmin = -torch.abs(q).max().detach().cpu().numpy(), vmax = torch.abs(q).max().detach().cpu().numpy())
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    cb = fig.colorbar(sc, cax=cax1,orientation='vertical')
    if label is not None:
        axis.set_title(label)
    if x.shape[1] > 1:
        square = patches.Rectangle((domainMin[0].detach().cpu().numpy(), domainMin[1].detach().cpu().numpy()), domainMax[0].detach().cpu().numpy() - domainMin[0].detach().cpu().numpy(), domainMax[1].detach().cpu().numpy() - domainMin[1].detach().cpu().numpy(), linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')
    axis.set_xlim(-1.05,1.05)
    axis.set_ylim(-1.05,1.05)
    return sc, cb

    # sc = axis.scatter(state['fluidPositions'][:,0], state['fluidPositions'][:,1], s = 8, c = q)
    # ax1_divider = make_axes_locatable(axis)
    # cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    # cb = fig.colorbar(sc, cax=cax1,orientation='vertical')
    # if label is not None:
    #     axis.set_title(label)
    # axis.set_aspect('equal')
    # axis.set_xlim(-1.05,1.05)
    # axis.set_ylim(-1.05,1.05)

def plotImplicitSDF(sdf, ngrid = 255, minExtent = -1, maxExtent = 1):
    ngrid = 255
    x = torch.linspace(minExtent, maxExtent, ngrid, dtype = torch.float32)
    y = torch.linspace(minExtent, maxExtent, ngrid, dtype = torch.float32)

    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    P = torch.stack([X,Y], dim=-1)
    points = P.reshape(-1,2)

    plotSDF(sdf(torch.clone(points),), X, Y, 2, 2)



def setPlotBaseAttributes(axis, config):
    domainMin = config['domain']['minExtent'].detach().cpu().numpy()
    domainMax = config['domain']['maxExtent'].detach().cpu().numpy()
    axis.set_xlim(domainMin[0], domainMax[0])
    axis.set_ylim(domainMin[1], domainMax[1])
    square = patches.Rectangle((domainMin[0], domainMin[1]), domainMax[0] - domainMin[0], domainMax[1] - domainMin[1], linewidth=1, edgecolor='b', facecolor='none',ls='--')
    axis.add_patch(square)
    axis.set_aspect('equal')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    # axis.set_xticklabels([])
    # axis.set_yticklabels([])

import matplotlib
from diffSPH.v2.sphOps import sphOperation
def mapToGrid(visualizationState, quantity):
    return sphOperation(
        (None, visualizationState['fluidMasses']), 
        (None, visualizationState['fluidDensities']), 
        (quantity, quantity), 
        (visualizationState['gridNeighborhood']['indices'][0], visualizationState['gridNeighborhood']['indices'][1]), visualizationState['gridNeighborhood']['kernels'], visualizationState['gridNeighborhood']['gradients'], 
        visualizationState['gridNeighborhood']['distances'], visualizationState['gridNeighborhood']['vectors'],
        visualizationState['gridNeighborhood']['supports'],   
        visualizationState['grid'].shape[0], operation = 'interpolate')

from diffSPH.v2.modules.neighborhood import neighborSearch, fluidNeighborSearch
from diffSPH.kernels import getKernel
from diffSPH.v2.sphOps import sphOperationFluidState, sphOperation


def prepVisualizationState(perennialState, config, nGrid = 128):
    visualizationState = {}

    visualizationState['fluidNeighborhood'] = fluidNeighborSearch(perennialState, config)
    visualizationState['fluidDensities'] = sphOperationFluidState(perennialState, None, 'density')
    visualizationState['fluidMasses'] = perennialState['fluidMasses']

    visualizationState['fluidVelocities'] = perennialState['fluidVelocities']
    x = perennialState['fluidPositions']

    periodicity = config['domain']['periodicity']
    minD = config['domain']['minExtent']
    maxD = config['domain']['maxExtent']

    if periodicity[0] and not periodicity[1]:
        visualizationState['fluidPositions'] = torch.stack((torch.remainder(x[:,0] - minD[0], maxD[0] - minD[0]) + minD[0], x[:,1]), dim = 1)
    elif not periodicity[0] and periodicity[1]:
        visualizationState['fluidPositions'] = torch.stack((x[:,0], torch.remainder(x[:,1] - minD[1], maxD[1] - minD[1]) + minD[1]), dim = 1)
    elif periodicity[0] and periodicity[1]:
        visualizationState['fluidPositions'] = torch.remainder(x - minD, maxD - minD) + minD
    else:
        visualizationState['fluidPositions'] = x  

    nGrid = 128
    xGrid = torch.linspace(config['domain']['minExtent'][0], config['domain']['maxExtent'][0], nGrid, dtype = perennialState['fluidPositions'].dtype, device = perennialState['fluidPositions'].device)
    yGrid = torch.linspace(config['domain']['minExtent'][1], config['domain']['maxExtent'][1], nGrid, dtype = perennialState['fluidPositions'].dtype, device = perennialState['fluidPositions'].device)
    X, Y = torch.meshgrid(xGrid, yGrid, indexing = 'xy')
    P = torch.stack([X,Y], dim=-1).flatten(0,1)

    visualizationState['grid'] = P
    visualizationState['X'] = X
    visualizationState['Y'] = Y
    visualizationState['nGrid'] = nGrid
    i, j, rij, xij, hij, Wij, gradWij = neighborSearch(visualizationState['grid'], visualizationState['fluidPositions'], 0, perennialState['fluidSupports'], getKernel('Wendland2'), config['domain']['dim'], config['domain']['periodicity'], config['domain']['minExtent'], config['domain']['maxExtent'], mode = 'scatter', algorithm ='compact')
    visualizationState['gridNeighborhood'] = {}
    visualizationState['gridNeighborhood']['indices'] = (i, j)
    visualizationState['gridNeighborhood']['distances'] = rij
    visualizationState['gridNeighborhood']['vectors'] = xij
    visualizationState['gridNeighborhood']['kernels'] = Wij
    visualizationState['gridNeighborhood']['gradients'] = gradWij
    visualizationState['gridNeighborhood']['supports'] = hij

    return visualizationState

def visualizeParticles(fig, axis, config, visualizationState, inputQuantity, mapping = '.x', cbar = True, cmap = 'viridis', scaling = 'symlog', s = 4, linthresh = 0.5, midPoint = 0, gridVisualization = False):        
    # print(inputQuantity.shape)
    setPlotBaseAttributes(axis, config)
    if len(inputQuantity.shape) == 2:
        # Non scalar quantity
        if mapping == '.x' or mapping == '[0]':
            quantity = inputQuantity[:,0]
        if mapping == '.y' or mapping == '[1]':
            quantity = inputQuantity[:,1]
        if mapping == '.z' or mapping == '[2]':
            quantity = inputQuantity[:,2]
        if mapping == '.w' or mapping == '[3]':
            quantity = inputQuantity[:,3]
        if mapping == 'Linf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = float('inf'))
        if mapping == 'L-inf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = -float('inf'))
        if mapping == 'L0':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 0)
        if mapping == 'L1':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 1)
        if mapping == 'L2' or mapping == 'norm' or mapping == 'magnitude':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 2)
        if mapping == 'theta':
            quantity = torch.atan2(inputQuantity[:,1], inputQuantity[:,0])
    else:
        quantity = inputQuantity

    pos_x = visualizationState['fluidPositions']

    minScale = torch.min(quantity)
    maxScale = torch.max(quantity)
    if 'sym' in scaling:
        minScale = - torch.max(torch.abs(quantity))
        maxScale =   torch.max(torch.abs(quantity))
        if 'log'in scaling:
            norm = matplotlib.colors.SymLogNorm(vmin = minScale, vmax = maxScale, linthresh = linthresh)
        else:
            minScale = - torch.max(torch.abs(quantity - midPoint))
            maxScale =   torch.max(torch.abs(quantity - midPoint))
            norm = matplotlib.colors.CenteredNorm(vcenter = midPoint, halfrange = maxScale)
    else:
        if 'log'in scaling:
            vmm = torch.min(torch.abs(quantity[quantity!= 0]))
            norm = matplotlib.colors.LogNorm(vmin = vmm, vmax = maxScale)
        else:
            norm = matplotlib.colors.Normalize(vmin = minScale, vmax = maxScale)
        
    if not gridVisualization:
        sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = quantity.detach().cpu().numpy(), cmap = cmap, norm = norm)
    else:
        gridDensity = mapToGrid(visualizationState, quantity)
        X = visualizationState['X']
        Y = visualizationState['Y']
        sc = axis.pcolormesh(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), gridDensity.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), cmap = cmap, norm = norm)
        
    if cbar:
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
        cb = fig.colorbar(sc, cax=cax1,orientation='vertical')
        cb.ax.tick_params(labelsize=8)
    # if periodicX:
    #     axis.axis('equal')
    #     axis.set_xlim(minDomain[0], maxDomain[0])
    #     axis.set_ylim(minDomain[1], maxDomain[1])
    # else:
    #     axis.set_aspect('equal', 'box')

    return {'plot': sc, 'cbar': cb if cbar else None, 'mapping': mapping, 'colormap': cmap, 'scale': scaling, 'size':4, 'mapToGrid': gridVisualization, 'midPoint' : midPoint, 'linthresh': linthresh}
    


def updatePlot(plotState, visualizationState, inputQuantity):        
    # print(inputQuantity.shape)
    # setPlotBaseAttributes(axis, config)
    mapping = plotState['mapping']
    scaling = plotState['scale']
    midPoint = plotState['midPoint']
    linthresh = plotState['linthresh']
    s = plotState['size']
    gridVisualization = plotState['mapToGrid']

    if len(inputQuantity.shape) == 2:
        # Non scalar quantity
        if mapping == '.x' or mapping == '[0]':
            quantity = inputQuantity[:,0]
        if mapping == '.y' or mapping == '[1]':
            quantity = inputQuantity[:,1]
        if mapping == '.z' or mapping == '[2]':
            quantity = inputQuantity[:,2]
        if mapping == '.w' or mapping == '[3]':
            quantity = inputQuantity[:,3]
        if mapping == 'Linf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = float('inf'))
        if mapping == 'L-inf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = -float('inf'))
        if mapping == 'L0':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 0)
        if mapping == 'L1':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 1)
        if mapping == 'L2' or mapping == 'norm' or mapping == 'magnitude':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 2)
        if mapping == 'theta':
            quantity = torch.atan2(inputQuantity[:,1], inputQuantity[:,0])
    else:
        quantity = inputQuantity

    pos_x = visualizationState['fluidPositions']
    qcpu = quantity.detach().cpu()
    minScale = torch.min(qcpu)
    maxScale = torch.max(qcpu)
    if 'sym' in scaling:
        minScale = - torch.max(torch.abs(qcpu))
        maxScale =   torch.max(torch.abs(qcpu))
        if 'log'in scaling:
            norm = matplotlib.colors.SymLogNorm(vmin = minScale, vmax = maxScale, linthresh = linthresh)
        else:
            minScale = - torch.max(torch.abs(qcpu - midPoint))
            maxScale =   torch.max(torch.abs(qcpu - midPoint))
            norm = matplotlib.colors.CenteredNorm(vcenter = midPoint, halfrange = maxScale)
    else:
        if 'log'in scaling:
            vmm = torch.min(torch.abs(qcpu[qcpu!= 0]))
            norm = matplotlib.colors.LogNorm(vmin = vmm, vmax = maxScale)
        else:
            norm = matplotlib.colors.Normalize(vmin = minScale, vmax = maxScale)
        
    if not gridVisualization:
        sc = plotState['plot']
        sc.set_offsets(pos_x.detach().cpu().numpy())
        sc.set_array(qcpu.numpy())
        sc.set_norm(norm)

        # scVelocity_x.set_clim(vmin = torch.abs(c).max().item() * -1, vmax = torch.abs(c).max().item())
        # cbarVelocity_x.update_normal(scVelocity_x)
        # sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = quantity.detach().cpu().numpy(), cmap = cmap, norm = norm)
    else:
        sc = plotState['plot']
        gridDensity = mapToGrid(visualizationState, quantity)
        sc.set_array(gridDensity.detach().cpu().numpy())
        sc.set_norm(norm)
        # sc = axis.pcolormesh(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), gridDensity.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), cmap = cmap, norm = norm)
    