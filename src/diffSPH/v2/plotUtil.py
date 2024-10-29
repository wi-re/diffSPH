from matplotlib import pyplot as plt
import numpy as np
import torch
import copy
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.math import scatter_sum
from diffSPH.v2.plotting import visualizeParticleQuantity, prepVisualizationState

from diffSPH.v2.sampling import sampleVelocityField

def plotPotentialField(perennialState, config, potential, rampedPotential, s = 1):
    fig, axis = plt.subplots(2,3, figsize = (12,7), squeeze = False)

        
    filteredState = copy.deepcopy(perennialState)
    filteredState['fluid']['potential'] = rampedPotential

    velocity, divergence = sampleVelocityField(filteredState['fluid'], perennialState['fluid']['neighborhood'])

    u_max = torch.linalg.norm(velocity, dim = 1).max() + 1e-6
    u_factor = config['fluid']['u_mag'] / u_max
    velocity = velocity * u_factor
    # velocity = velocity / (torch.linalg.norm(velocity, dim = 1, keepdim = True) + 1e-6) * config['fluid']['u_mag']

    divergence = sphOperationStates(perennialState['fluid'], perennialState['fluid'], (velocity, velocity), operation = 'divergence', neighborhood = perennialState['fluid']['neighborhood'])

    visualizationState = prepVisualizationState(perennialState, config, grid = True)
    # plotRegions(regions, axis[0,0])
    # plotRegions(regions, axis[0,1])

    indexPlot = visualizeParticleQuantity(fig, axis[0,0], config, visualizationState, quantity =
                                        potential,
                                        mapping = '.x', s = s, 
                            scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'Potential', which = 'fluid', plotBoth = False, midPoint = 0)
    quantPlot = visualizeParticleQuantity(fig, axis[0,1], config, visualizationState, quantity = 
                                        #   'velocities', 
                                        rampedPotential,
                                        mapping = '.y', s = s, 
                            scaling = 'sym', gridVisualization=False, cmap = 'icefire', streamLines = False, operation = None, title = 'Ramped', plotBoth = False, which = 'fluid', midPoint = 0)
    divergencePlot = visualizeParticleQuantity(fig, axis[0,2], config, visualizationState, quantity = 
                                            #    rampDivergenceFree(perennialState['fluid']['positions'], perennialState['fluid']['potential'], boundary_sdf, 0, d0 = 0.25), 
                                            rampedPotential / (potential + 1e-6),
                                            # perennialState['fluid']['potential'],
                                        #    divergence,
                                            mapping = '', s = s,
                                scaling = 'sym', gridVisualization=False, cmap = 'icefire', streamLines = False, operation = None, title = 'Ramp', plotBoth = False, which = 'fluid', midPoint = 0)

    indexPlot = visualizeParticleQuantity(fig, axis[1,0], config, visualizationState, quantity =
                                        velocity,
                                        mapping = '.x', s = s, 
                            scaling = 'sym', gridVisualization=False, cmap = 'RdBu_r', title = 'Particle x-Velocity', which = 'fluid', plotBoth = False, midPoint = 0)
    quantPlot = visualizeParticleQuantity(fig, axis[1,1], config, visualizationState, quantity = 
                                        velocity, 
                                        mapping = '.y', s = s, 
                            scaling = 'sym', gridVisualization=False, cmap = 'RdBu_r', streamLines = False, operation = None, title = 'Particle y-Velocity', plotBoth = False, which = 'fluid', midPoint = 0)

    # divergencePlot = visualizeParticleQuantity(fig, axis[1,2], config, visualizationState, quantity = 
    #                                        divergence,
    #                                            mapping = '', s = s,
    #                             scaling = 'sym', gridVisualization=False, cmap = 'coolwarm', streamLines = False, operation = None, title = 'Divergence', plotBoth = False, which = 'fluid', midPoint = 0)

    divergencePlot = visualizeParticleQuantity(fig, axis[1,2], config, visualizationState, quantity = 
                                        velocity,
                                            mapping = 'L2', s = s,
                                scaling = 'lin', gridVisualization=True, cmap = 'viridis', streamLines = True, operation = None, title = 'Velocity Magnitude', plotBoth = False, which = 'fluid', midPoint = 0)


    fig.tight_layout()

    fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"] + perennialState["boundary"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid + {perennialState["boundary"]["numParticles"]} boundary)')
    fig.tight_layout()

from diffSPH.v2.plotting import setPlotBaseAttributes, plotRegions
def plotInitialParticleSampling(perennialState, config):
    regions = config['regions']
    fig, axis = plt.subplots(1, 3, figsize = (15,5), squeeze = False)

    plotRegions(regions, axis[0,0], plotParticles=False)
    setPlotBaseAttributes(axis[0,0], config)
    axis[0,0].scatter(perennialState['fluid']['positions'][:,0].cpu(), perennialState['fluid']['positions'][:,1].cpu(), s = 1, c = 'b')

    plotRegions(regions, axis[0,1], plotParticles=False)
    setPlotBaseAttributes(axis[0,1], config)
    axis[0,1].scatter(perennialState['boundary']['positions'][:,0].cpu(), perennialState['boundary']['positions'][:,1].cpu(), s = 1, c = 'r')

    plotRegions(regions, axis[0,2], plotParticles=False)
    setPlotBaseAttributes(axis[0,2], config)
    axis[0,2].scatter(perennialState['fluid']['positions'][:,0].cpu(), perennialState['fluid']['positions'][:,1].cpu(), s = 1, c = 'b')
    axis[0,2].scatter(perennialState['boundary']['positions'][:,0].cpu(), perennialState['boundary']['positions'][:,1].cpu(), s = 1, c = 'r')

    fig.tight_layout()



from diffSPH.v2.plotting import setPlotBaseAttributes
from diffSPH.v2.plotting import plotSDF
from diffSPH.v2.finiteDifference import computeGradient
from mpl_toolkits.axes_grid1 import make_axes_locatable
from diffSPH.v2.sdf import getSDF, sdfFunctions, operatorDict


def plotRegionswSDF(config):
    regions = config['regions']

    fig, axis = plt.subplots(1,2, figsize = (15,5), squeeze = False)
    setPlotBaseAttributes(axis[0,0], config)
    # # setPlotBaseAttributes(axis[0,1], config)

    # # plotRegions(regions, axis[0,0])
    plotRegions(regions, axis[0,0])
    if config['boundary']['active']:
        boundary_sdfs = [region['sdf'] for region in regions if region['type'] == 'boundary']
        combined_sdf = boundary_sdfs[0]
        for sdf in boundary_sdfs[1:]:
            combined_sdf = operatorDict['union'](combined_sdf, sdf)
        ngrid = 256
        x = torch.linspace(config['domain']['minExtent'][0], config['domain']['maxExtent'][0], ngrid, dtype = torch.float32)
        y = torch.linspace(config['domain']['minExtent'][1], config['domain']['maxExtent'][1], ngrid, dtype = torch.float32)
        X_, Y_ = torch.meshgrid(x, y, indexing = 'ij')
        P = torch.stack([X_,Y_], dim=-1)
        points = P.reshape(-1,2)

        fx_ = combined_sdf(torch.clone(points),)

        X = X_.detach().cpu()
        Y = Y_.detach().cpu()
        fx = fx_.detach().cpu()

        fx = fx.reshape(256, 256)

        domainExtent = config['domain']['maxExtent'] - config['domain']['minExtent']
        extent = domainExtent.max().item()

        setPlotBaseAttributes(axis[0,1], config)
        # # setPlotBaseAttributes(axis[0,1], config)

        # # plotRegions(regions, axis[0,0])
        plotRegions(regions, axis[0,1], plotParticles=False)
        output = computeGradient(fx, extent, config['domain']['dim'], 1)
        # Plot the isocontours of fx
        axis[0,1].contour(X.numpy(), Y.numpy(), fx.numpy(), levels=[0], colors='black')


        # sdfValues = sdCircle(P, 1.0)
        # sdfGradient = gradient(sdfValues, 4, 1, 2)
        # sdfGradient = torch.stack([centralDifference(sdfValues, 1/ngrid, 1, 4), centralDifference(sdfValues, 1/ngrid, 1, 4)], axis=-1)
        # sdfValues = fx
        # sdfGradient = output
        # print(sdfGradient.shape)

        im = axis[0,1].pcolormesh(X.numpy(), Y.numpy(), fx.numpy(), cmap='Spectral',vmin = - torch.max(torch.abs(fx)).numpy(), vmax = torch.max(torch.abs(fx)).numpy())
        axis[0,1].set_title("SDF")
        axis[0,1].set_aspect('equal', 'box')
        divider = make_axes_locatable(axis[0,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        spacing = 16
        axis[0,1].quiver(X[::spacing,::spacing].numpy(), Y[::spacing,::spacing].numpy(), output[::spacing,::spacing,0].numpy(), output[::spacing,::spacing,1].numpy())

        # plotSDF(, X, Y, 2, 2)
        fig.tight_layout()


from diffSPH.v2.plotting import setPlotBaseAttributes
from diffSPH.v2.modules.mDBC import buildBoundaryGhostParticles

def plotBoundaryParticleswGhosts(perennialState, config):

    fig, axis = plt.subplots(2,3, figsize = (12,8), squeeze = False)

    for ax in axis.flatten():
        setPlotBaseAttributes(ax, config)

    p = perennialState['boundary']['positions'].detach().cpu().numpy()

    ghost =  buildBoundaryGhostParticles(perennialState, config)
    g = ghost['positions'].detach().cpu().numpy() 

    sc = axis[0,0].scatter(p[:,0], p[:,1], s = 1, c = perennialState['boundary']['distances'].detach().cpu().numpy())
    # sc = axis[0,0].scatter(p[:,0], p[:,1], s = 1, c = sdfValues.detach().cpu().numpy())
    fig.colorbar(sc, ax = axis[0,0])
    sc = axis[0,1].scatter(p[:,0], p[:,1], s = 1, c = perennialState['boundary']['normals'][:,0].detach().cpu().numpy())
    # sc = axis[0,1].scatter(p[:,0], p[:,1], s = 1, c = sdfGradients[:,0].detach().cpu().numpy())
    fig.colorbar(sc, ax = axis[0,1])
    sc = axis[0,2].scatter(p[:,0], p[:,1], s = 1, c = perennialState['boundary']['normals'][:,1].detach().cpu().numpy())
    # sc = axis[0,2].scatter(p[:,0], p[:,1], s = 1, c = sdfGradients[:,1].detach().cpu().numpy())
    fig.colorbar(sc, ax = axis[0,2])


    axis[1,0].scatter(p[:,0], p[:,1], s = 1, c = np.arange(p.shape[0]))
    axis[1,1].scatter(g[:,0], g[:,1], s = 1, c = np.arange(p.shape[0]))

    axis[1,2].scatter(p[:,0], p[:,1], s = 1, c = np.arange(p.shape[0]))
    axis[1,2].scatter(g[:,0], g[:,1], s = 1, c = np.arange(p.shape[0]))

    fig.tight_layout()


from diffSPH.v2.modules.mDBC import buildBoundaryGhostParticles
from diffSPH.v2.sphOps import sphOperationStates, LiuLiuConsistent
from diffSPH.v2.modules.mDBC import mDBCDensity
from diffSPH.v2.modules.boundaryViscosity import computeBoundaryVelocities

def plotBoundaryVelocityFields(perennialState, config):

    fig, axis = plt.subplots(3,3, figsize = (12,12), squeeze = False, sharex=True, sharey=True)

    # perennialState['fluid']['velocities'][:,0] = perennialState['fluid']['positions'][:,0]
    # perennialState['fluid']['velocities'][:,1] = perennialState['fluid']['positions'][:,1]

    # perennialState['fluid']['velocities'][:,0] = 1
    # perennialState['fluid']['velocities'][:,1] = 1

    # perennialState['fluid']['densities'] = 1000 + perennialState['fluid']['positions'][:,0]**2 * 2000
    # perennialState['fluid']['densities'][:] = 1000

    perennialState['boundaryGhost'] = buildBoundaryGhostParticles(perennialState, config)
    boundaryDensity, shepardDensity = mDBCDensity(perennialState, config)
    perennialState['boundary']['densities'][:] = 1000#boundaryDensity

    # plotRegions(regions, axis[0,0])
    # plotRegions(regions, axis[0,1])

    s = 4



    ghostState = buildBoundaryGhostParticles(perennialState, config)
    # display(ghostState)
    perennialState['boundaryGhost'] = buildBoundaryGhostParticles(perennialState, config)



    # ghostNormal = -2 * perennialState['boundary']['distances'].view(-1,1) * perennialState['boundary']['normals'] 
    # solution, M, b = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,0])
    # perennialState['boundary']['velocities'][:,0] = solution[:,0] + torch.einsum('nd, nd -> n', ghostNormal, solution[:,1:])
    # solution_uy, M_uy, b_uy = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,1])
    # perennialState['boundary']['velocities'][:,1] = solution_uy[:,0] + torch.einsum('nd, nd -> n', ghostNormal, solution_uy[:,1:])
    perennialState['boundary']['velocities'] = computeBoundaryVelocities(perennialState, config)

    # perennialState['boundary']['velocities'][:,0] = 0

    visualizationState = prepVisualizationState(perennialState, config, grid = True)

    fluid_ux = visualizeParticleQuantity(fig, axis[0,0], config, visualizationState, quantity = 'velocities', mapping = '.x', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'x-Velocity', which = 'fluid', plotBoth = False, midPoint = 0)
    fluid_uy = visualizeParticleQuantity(fig, axis[0,1], config, visualizationState, quantity = 'velocities', mapping = '.y', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'y-Velocity', which = 'fluid', plotBoth = False, midPoint = 0)
    fluid_umag = visualizeParticleQuantity(fig, axis[0,2], config, visualizationState, quantity = 'velocities', mapping = 'L2', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'Velocity', which = 'fluid', plotBoth = False, midPoint = 0)


    boundary_ux = visualizeParticleQuantity(fig, axis[1,0], config, visualizationState, quantity = 'velocities', mapping = '.x', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'x-Velocity', which = 'boundary', plotBoth = False, midPoint = 0)
    boundary_uy = visualizeParticleQuantity(fig, axis[1,1], config, visualizationState, quantity = 'velocities', mapping = '.y', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'y-Velocity', which = 'boundary', plotBoth = False, midPoint = 0)
    boundary_umag = visualizeParticleQuantity(fig, axis[1,2], config, visualizationState, quantity = 'velocities', mapping = 'L2', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'Velocity', which = 'boundary', plotBoth = False, midPoint = 0)

    both_ux = visualizeParticleQuantity(fig, axis[2,0], config, visualizationState, quantity = 'velocities', mapping = '.x', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'x-Velocity', which = 'both', plotBoth = True, midPoint = 0)
    both_uy = visualizeParticleQuantity(fig, axis[2,1], config, visualizationState, quantity = 'velocities', mapping = '.y', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'y-Velocity', which = 'both', plotBoth = True, midPoint = 0)
    both_umag = visualizeParticleQuantity(fig, axis[2,2], config, visualizationState, quantity = 'velocities', mapping = 'L2', s = s, scaling = 'sym', gridVisualization=False, cmap = 'icefire', title = 'Velocity', which = 'both', plotBoth = True, midPoint = 0)
                                        
    fig.tight_layout()
