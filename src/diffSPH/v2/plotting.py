from diffSPH.v2.finiteDifference import computeGradient
from diffSPH.v2.sdf import getSDF, sdfFunctions, operatorDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch


def plotSDF(fx_, X_, Y_, extent, dim, ngrid = 255):
    fig, axis = plt.subplots(1, 3, figsize=(14,4), sharex = False, sharey = False, squeeze = False)

    X = X_.detach().cpu()
    Y = Y_.detach().cpu()
    fx = fx_.detach().cpu()

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
import copy

def prepVisualizationState(perennialState, config, nGrid = 128):
    visualizationState = copy.deepcopy(perennialState)

    visualizationState['fluidNeighborhood'] = fluidNeighborSearch(visualizationState, config)
    visualizationState['fluidDensities'] = sphOperationFluidState(visualizationState, None, 'density')
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

    # nGrid = 128
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
    

from diffSPH.v2.plotting import mapToGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy.stats as stats
from scipy.signal import find_peaks_cwt
import numpy as np

def computePSD(perennialState, grid_umag, config, nGrid = 255):

    visualizationState = prepVisualizationState(perennialState, config, nGrid = 255)
    X = visualizationState['X']
    Y = visualizationState['Y']

    grid_ux = mapToGrid(visualizationState, grid_umag).reshape(X.shape)
    # grid_uy = mapToGrid(visualizationState, perennialState['fluidVelocities'][:,1]).reshape(X.shape)
    # grid_umag = (grid_ux**2 + grid_uy**2)

    data = grid_ux.detach().cpu().numpy()

    # Normalize your grid data to span -1 to 1
    grid_umag_normalized = 2 * (data - np.min(data)) / np.ptp(data) - 1

    # Compute the 2D FFT
    fft_grid_umag = np.fft.fft2(grid_umag_normalized)

    # Shift the zero frequency component to the center of the spectrum
    fft_shifted = np.fft.fftshift(fft_grid_umag)

    # Get the frequencies for the x and y axis
    freqs_x = np.fft.fftshift(np.fft.fftfreq(data.shape[0]))
    freqs_y = np.fft.fftshift(np.fft.fftfreq(data.shape[1]))

    delta_x = (config['domain']['maxExtent'] - config['domain']['minExtent']).detach().cpu().numpy()[0] / data.shape[0]
    delta_y = (config['domain']['maxExtent'] - config['domain']['minExtent']).detach().cpu().numpy()[0] / data.shape[1]
    physical_freqs_x = freqs_x / delta_x
    physical_freqs_y = freqs_y / delta_y

    # Create a grid of frequencies
    freqs_xx, freqs_yy = np.meshgrid(freqs_x, freqs_y, indexing = 'xy')
    physical_freqs_xx, physical_freqs_yy = np.meshgrid(physical_freqs_x, physical_freqs_y, indexing = 'xy')

    # Get the magnitude of the FFT
    fft_mag = np.abs(fft_shifted)

    knrm = np.sqrt(physical_freqs_xx**2 + physical_freqs_yy**2)
    fourier_amplitudes = np.abs(fft_mag)**2

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # print(knrm.shape, fourier_amplitudes.shape)

    kbins = np.linspace(knrm.min(), knrm.max(), fft_mag.shape[0]//2)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    # Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    # print(Abins.shape)
    peaks = find_peaks_cwt(Abins, np.arange(1, 10))
    return visualizationState, physical_freqs_xx, physical_freqs_yy, fft_mag, kvals, Abins, peaks


    fig, axis = plt.subplots(1,1, figsize = (6,6))
    axis.loglog(kvals, Abins)
    axis.set_xlabel("$k$")
    axis.set_ylabel("$P(k)$")
    axis.axvline(kvals[np.argmax(Abins)], c = 'black', linestyle = '-')

    print(peaks)
    # Get the seaborn color palette
    color_palette = sns.color_palette()



    fig, axis = plt.subplots(1,1, figsize = (6,6))
    axis.loglog(kvals, Abins)
    axis.set_xlabel("$k$")
    axis.set_ylabel("$P(k)$")
    # axis.axvline(kvals[np.argmax(Abins)], c = 'black', linestyle = '-')
    # axis.text(kvals[np.argmax(Abins)], 0.5*10**-5, f'k = {kvals[np.argmax(Abins)]:.2f}', ha = 'right', va = 'bottom', rotation = 90)

    for i, peak in enumerate(peaks):
        axis.axvline(kvals[peak ], c = color_palette[i], linestyle = '--')
        axis.text(kvals[peak ], 0.5*10**-5, f'k = {kvals[peak]:.2f}', ha = 'right', va = 'bottom', rotation = 90, c = color_palette[i])

    axis.text(kvals[np.argmax(Abins)], 0.5*10**-5, f'k = {kvals[np.argmax(Abins)]:.2f}', ha = 'right', va = 'bottom', rotation = 90)

    fig.tight_layout()

def plotFFT(fig, axis, fx, fy, fft_data):
    logData = np.log(fft_data)
    im = axis.pcolormesh(fx, fy, logData, shading='auto', vmin = -np.max(np.abs(logData)), vmax = np.max(np.abs(logData)), cmap = 'Spectral_r')
    axis.set_xlabel('Frequency (x)')
    axis.set_ylabel('Frequency (y)')
    axis.set_title('FFT Magnitude')
    cax = make_axes_locatable(axis).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
def plotPSD(fig, axis, kvals, Abins, peaks = None):
    axis.loglog(kvals, Abins)
    axis.set_xlabel("$k$")
    axis.set_ylabel("$P(k)$")
    if peaks is not None:
        color_palette = sns.color_palette()
        for i, peak in enumerate(peaks):
            axis.axvline(kvals[peak ], c = color_palette[i % len(color_palette)], linestyle = '--')
            axis.text(kvals[peak ], 0.5*10**-5, f'k = {kvals[peak]:.2f}', ha = 'right', va = 'bottom', rotation = 90, c = color_palette[i% len(color_palette)])
