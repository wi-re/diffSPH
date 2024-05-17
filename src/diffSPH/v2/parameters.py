from diffSPH.parameter import Parameter
import torch
from typing import Union, List
from diffSPH.kernels import getKernel
from diffSPH.v2.sampling import optimizeArea
from diffSPH.v2.math import volumeToSupport
import numpy as np

particleParameters = [
    Parameter('particle', 'nx', int, 64, required = False, export = True),
    Parameter('particle', 'dx', float, -1.0, required = False, export = True),
]

fluidParameters = [
    Parameter('fluid', 'rho0', float, 1000, required = False, export = True),
    Parameter('fluid', 'mu', float, 0.1, required = False, export = True),
    Parameter('fluid', 'cs', float, 10, required = False, export = True),
]

simulationParameters = [
    Parameter('simulation', 'correctArea', bool, False, required = False, export = True),
    Parameter('simulation', 'supportScheme', str, 'symmetric', required = False, export = True),
    Parameter('simulation', 'densityScheme', str, 'continuity', required = False, export = True),
]

integrationParameters = [
]

computeParameters = [
    Parameter('compute', 'device', str, 'cpu', required = False, export = False),
    Parameter('compute', 'precision', str, 'float32', required = False, export = False),
]

kernelParameters = [
    Parameter('kernel', 'name', str, 'Wendland2', required = False, export = True),
    Parameter('kernel', 'targetNeighbors', int, -1, required = False, export = True),
    # Parameter('kernel', 'function', torch.nn.Module, None, required = False, export = False),
]

domainParameters = [
    Parameter('domain', 'minExtent', Union[float, List[float]], -1, required = False, export = True),
    Parameter('domain', 'maxExtent', Union[float, List[float]], 1, required = False, export = True),
    Parameter('domain', 'dim', int, 2, required = False, export = False),
    Parameter('domain', 'periodic', Union[float, List[bool]], False, required = False, export = True),
]

import random
noiseParameters = [
    Parameter('noise', 'n', int, 64, required = False, export = True),
    Parameter('noise', 'baseFrequency', float, 1, required = False, export = True),
    Parameter('noise', 'dim', int, 2, required = False, export = True),
    Parameter('noise', 'octaves', int, 4, required = False, export = True),
    Parameter('noise', 'persistence', float, 0.5, required = False, export = True),
    Parameter('noise', 'lacunarity', float, 2, required = False, export = True),
    Parameter('noise', 'seed', int, random.randint(0,77777777), required = False, export = True),
    Parameter('noise', 'tileable', bool, True, required = False, export = True),
    Parameter('noise', 'kind', str, 'simplex', required = False, export = True),
]

plotStateDict = [
    Parameter('', 'val', str, 'fluidIndex', required = False, export = True, hint='Value to plot'),
    Parameter('', 'cbar', bool, True, required = False, export = True, hint='Colorbar'),
    Parameter('', 'cmap', str, 'twilight', required = False, export = True, hint='Colormap'),
    Parameter('', 'scale', str, 'lin', required = False, export = True, hint='Scale'),
    Parameter('', 'size', int, 5, required = False, export = True, hint='Size'),
    Parameter('', 'gridVis', bool, False, required = False, export = True, hint='Grid Mapping'),
    Parameter('', 'title', str, 'Fluid Index', required = False, export = True, hint='Title'),
    Parameter('', 'midPoint', float, 0, required = False, export = True, hint='Midpoint'),
    Parameter('', 'mapping', str, '.x', required = False, export = True, hint='Mapping'),
]

boundaryParameters = [
    Parameter('boundary', 'active', bool, False, required = False, export = True, hint='Boundary condition active'),
    Parameter('boundary', 'type', str, 'mDBC', required = False, export = True, hint='Boundary condition type'),
    Parameter('boundary', 'static', bool, True, required = False, export = True, hint='Static boundary condition'),
    Parameter('boundary', 'twoWayCoupled', bool, False, required = False, export = True, hint='Two-way coupled boundary condition'),
]

plotParameters = [
    Parameter('plot', 'mosaic', str, 'A', required = False, export = True, hint='Mosaic plot layout'),
    Parameter('plot', 'figSize', List[float], [6,5.5], required = False, export = True, hint='Figure size'),
    Parameter('plot', 'export', bool, True, required = False, export = True, hint='Export plots'),
    Parameter('plot', 'updateInterval', int, 32, required = False, export = True, hint='Export interval'),
    Parameter('plot', 'namingScheme', str, 'timestep', required = False, export = True, hint='Naming scheme'),
    Parameter('plot', 'exportPath', str, 'output', required = False, export = True, hint='Export path'),
    Parameter('plot', 'gif', bool, True, required = False, export = True, hint='Export gif'),
    Parameter('plot', 'namingScheme', str, 'timestep', required = False, export = True, hint='Naming scheme'),
    Parameter('plot', 'plots', dict, {'A': {'val': 'fluidIndex', 'cbar': True, 'cmap': 'twilight', 'scale': 'lin', 'size': 5, 'gridVis' : False, 'title': 'Fluid Index'}},required = False, export = True, hint='Plots'),
    Parameter('plot', 'gifScale', int, 640, required = False, export = True, hint='Gif Output Size'),
    Parameter('plot', 'fps', float, -1, required = False, export = True, hint='Export FPS'),
    Parameter('plot', 'exportFPS', float, 30, required = False, export = True, hint='Export FPS'),

]

exportParameters = [
    Parameter('export', 'active', bool, True, required = False, export = True, hint='Export data'),
    Parameter('export', 'exportPath', str, 'export', required = False, export = True, hint='Export path'),
    Parameter('export', 'namingScheme', str, 'timestep', required = False, export = True, hint='Naming scheme'),
    Parameter('export', 'interval', int, 1, required = False, export = True, hint='Export interval'),
]

defaultParameters = particleParameters + computeParameters + domainParameters + kernelParameters + integrationParameters + simulationParameters + fluidParameters + plotParameters + exportParameters + noiseParameters + boundaryParameters

def parseComputeInfo(config):
    device = config['compute']['device']
    precision = config['compute']['precision']
    if precision == 'float32':
        dtype = torch.float32
    elif precision == 'float64':
        dtype = torch.float64
    else:
        raise ValueError('Precision not recognized')
    return device, dtype

def parseDomainConfig(config: dict):
    minExtent = config['domain']['minExtent']
    maxExtent = config['domain']['maxExtent']

    if isinstance(minExtent, list):
        minExtent = torch.tensor(minExtent, dtype = config['compute']['dtype'], device = config['compute']['device'])
    else:
        minExtent = torch.tensor([minExtent]*config['domain']['dim'], dtype = config['compute']['dtype'], device = config['compute']['device'])
    if isinstance(maxExtent, list):
        maxExtent = torch.tensor(maxExtent, dtype = config['compute']['dtype'], device = config['compute']['device'])
    else:
        maxExtent = torch.tensor([maxExtent]*config['domain']['dim'], dtype = config['compute']['dtype'], device = config['compute']['device'])
    
    if isinstance(config['domain']['periodic'], bool):
        periodicity = torch.tensor([config['domain']['periodic']] * config['domain']['dim'], device = config['compute']['device'], dtype = torch.bool)
    else:
        periodicity = torch.tensor(config['domain']['periodic'], device = config['compute']['device'], dtype = torch.bool)

    return minExtent, maxExtent, periodicity

def parseKernelConfig(config: dict):
    kernel = getKernel(config['kernel']['name'])
    targetNeighbors = config['kernel']['targetNeighbors']
    return kernel, targetNeighbors, kernel.kernelScale(config['domain']['dim'])

import datetime
def parseParticleConfig(config: dict):
    nx = config['particle']['nx']

    domainExtent = config['domain']['maxExtent'] - config['domain']['minExtent']
    if config['particle']['dx'] < 0.:
        dx = domainExtent / nx
        dx = torch.min(dx)
    else:
        dx = torch.tensor(config['particle']['dx'], dtype = config['compute']['dtype'], device = config['compute']['device'])
    volume = dx**config['domain']['dim']

    if config['kernel']['targetNeighbors'] == -1:
        h = 2 * dx * config['kernel']['kernelScale']
        if config['domain']['dim'] == 1:
            numNeighbors = 2 * h / dx
        elif config['domain']['dim'] == 2:
            numNeighbors = np.pi * h**2 / dx**2
        elif config['domain']['dim'] == 3:
            numNeighbors = 4/3 * np.pi * h**3 / dx**3
        config['kernel']['targetNeighbors'] = numNeighbors
        config['particle']['support'] = h
    else:
        h = volumeToSupport(volume, config['kernel']['targetNeighbors'], config['domain']['dim'])
        config['particle']['kernelScale'] = h / (2 * dx)

    if config['simulation']['correctArea']:
        optimizedVolume, *_ = optimizeArea(volume.item(), dx, volume.dtype, 'cpu', config['kernel']['targetNeighbors'], config['kernel']['function'], dim = config['domain']['dim'], thresh = 1e-7**2, maxIter = 64)
        optimizedSupport = volumeToSupport(optimizedVolume, config['kernel']['targetNeighbors'], config['domain']['dim'])
    else:
        optimizedVolume = volume
        optimizedSupport = h
    return dx, volume, h, optimizedVolume, optimizedSupport

def getDefaultParameters():
    return defaultParameters

def printDefaultParameters():
    for parameter in defaultParameters:
        print(parameter)

def parseDefaultParameters(config):
    # for parameter in defaultParameters:
        # print(parameter)

    for parameter in defaultParameters:
        parameter.parseConfig(config)

    config['compute']['device'], config['compute']['dtype'] = parseComputeInfo(config)
    config['domain']['minExtent'], config['domain']['maxExtent'], config['domain']['periodicity'] = parseDomainConfig(config)
    config['kernel']['function'], config['kernel']['targetNeighbors'], config['kernel']['kernelScale'] = parseKernelConfig(config)
    config['particle']['dx'], config['particle']['defaultVolume'], config['particle']['defaultSupport'], \
        config['particle']['volume'], config['particle']['support'] = parseParticleConfig(config)
    config['kernel']['kernelScale'] = config['particle']['support'] / (2 * config['particle']['dx'])
    config['particle']['smoothingLength'] = 2 * config['particle']['dx']
    config['noise']['n'] = config['particle']['nx']

    # for plot in config['plot']['plots']:
        # for parameter in plotStateDict:
            # parameter.parseConfig(config['plot']['plots'][plot])

    config['simulation']['timestamp'] = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    return config
    # print(config)
    

from diffSPH.v2.modules.viscosity import computeViscosityParameter
from diffSPH.v2.moduleWrapper import modules
def parseModuleParameters(config):
    for module in modules:
        params = module.getParameters()
        for param in params:
            param.parseConfig(config)
    config['diffusion']['nu_sph'] = computeViscosityParameter(None, config)
    return config