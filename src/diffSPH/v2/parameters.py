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

defaultParameters = particleParameters + computeParameters + domainParameters + kernelParameters + integrationParameters + simulationParameters + fluidParameters

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
        maxExtent = torch.tensor(minExtent, dtype = config['compute']['dtype'], device = config['compute']['device'])
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
    return config
    # print(config)
    


from diffSPH.v2.moduleWrapper import modules
def parseModuleParameters(config):
    for module in modules:
        params = module.getParameters()
        for param in params:
            param.parseConfig(config)
    return config