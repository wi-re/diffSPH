
import torch
import torch
from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum

from torch.profiler import record_function

# def computeViscosityMonaghan(simulationState, config):
#     with record_function("Viscosity [Monaghan]"):       
#         return config['fluid']['mu'] * sphOperationFluidState(simulationState, (simulationState['fluidVelocities'], simulationState['fluidVelocities']), operation = 'laplacian', gradientMode='conserving')# / simulationState['fluidDensities'].view(-1,1)

# simulationState['viscosity'] = physicalParameters['nu'] * sphOperation(
#     (areas, areas), 
#     (simulationState['rho'], simulationState['rho']),
#     (u, u),
#     (i, j), kernel, gradKernel, rij, xij, hij, x.shape[0], operation = 'laplacian', gradientMode = 'conserving') / simulationState['rho'].view(-1,1)

from diffSPH.v2.math import scatter_sum
def computeViscosityMonaghan1983(fluidState, config):
    eps = config['diffusion']['eps']

    (i,j) = fluidState['fluidNeighborhood']['indices']
    h_ij = fluidState['fluidNeighborhood']['supports']
    c_ij = config['fluid']['cs']
    rho_ij = (fluidState['fluidDensities'][i] + fluidState['fluidDensities'][j]) / 2
    alpha = config['diffusion']['alpha']

    nu = alpha * c_ij * h_ij / rho_ij

    v_ij = fluidState['fluidVelocities'][j] - fluidState['fluidVelocities'][i]
    r_ij = fluidState['fluidNeighborhood']['distances'] * h_ij
    x_ij = fluidState['fluidNeighborhood']['vectors']# * r_ij.view(-1,1)

    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

    pi_ij = -nu * vr_ij / (r_ij + eps * h_ij**2)
    if config['diffusion']['pi-switch']:
        pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

    return scatter_sum((fluidState['fluidMasses'][j] * pi_ij).view(-1,1) * fluidState['fluidNeighborhood']['gradients'], i, dim = 0, dim_size = fluidState['numParticles'])

def computeViscosityMonaghan1992(fluidState, config):
    eps = config['diffusion']['eps']

    (i,j) = fluidState['fluidNeighborhood']['indices']
    h_ij = fluidState['fluidNeighborhood']['supports']
    v_ij = fluidState['fluidVelocities'][j] - fluidState['fluidVelocities'][i]
    r_ij = fluidState['fluidNeighborhood']['distances'] * h_ij
    x_ij = fluidState['fluidNeighborhood']['vectors']# * r_ij.view(-1,1)

    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

    c_ij = config['fluid']['cs']
    rho_ij = (fluidState['fluidDensities'][i] + fluidState['fluidDensities'][j]) / 2
    alpha = config['diffusion']['alpha']
    beta = config['diffusion']['beta']

    nu = h_ij / rho_ij * (alpha * c_ij - beta *(h_ij * vr_ij / (r_ij + eps * h_ij**2)))

    if config['diffusion']['pi-switch']:
        pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

    pi_ij = -nu * vr_ij / (r_ij + eps * h_ij**2)
    return scatter_sum((fluidState['fluidMasses'][j] * pi_ij).view(-1,1) * fluidState['fluidNeighborhood']['gradients'], i, dim = 0, dim_size = fluidState['numParticles'])
    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)


    pass
def computeViscosityXSPH(fluidState, config):
    alpha = config['diffusion']['artificial-constant']

    (i,j) = fluidState['fluidNeighborhood']['indices']
    return alpha * sphOperationFluidState(fluidState, fluidState['fluidVelocities'][j] - fluidState['fluidVelocities'][i], 'interpolate') / config['timestep']['dt']
    pass

def computeViscosityNaive(fluidState, config):
    alpha = config['diffusion']['nu']
    return alpha * sphOperationFluidState(fluidState, (fluidState['fluidVelocities'], fluidState['fluidVelocities']), 'laplacian', 'conserving')

# From Price 2012: Smoothed particle hydrodynamics and magnetohydrodynamics [https://doi.org/10.1016/j.jcp.2010.12.011]
def computeViscosityPrice2012(fluidState, config):
    eps = config['diffusion']['eps']
    alpha = config['diffusion']['alpha']
    beta = config['diffusion']['beta']  

    (i,j) = fluidState['fluidNeighborhood']['indices']
    h_ij = fluidState['fluidNeighborhood']['supports']
    v_ij = fluidState['fluidVelocities'][i] - fluidState['fluidVelocities'][j]
    r_ij = fluidState['fluidNeighborhood']['distances'] * h_ij
    x_ij = fluidState['fluidNeighborhood']['vectors']# * r_ij.view(-1,1)
    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

    mu_ij = h_ij * vr_ij / (r_ij + eps * h_ij**2)

    c_ij = config['fluid']['cs']
    rho_ij = (fluidState['fluidDensities'][i] + fluidState['fluidDensities'][j]) / 2
    pi_ij = (-alpha * c_ij * mu_ij + beta  * mu_ij**2 ) / rho_ij

    if config['diffusion']['pi-switch']:
        pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

    # print(mu_ij, pi_ij)
    return -scatter_sum((fluidState['fluidMasses'][j] * pi_ij).view(-1,1) * fluidState['fluidNeighborhood']['gradients'], i, dim = 0, dim_size = fluidState['numParticles'])
    pass
def computeViscosityDeltaSPH_inviscid(fluidState, config):
    eps = config['diffusion']['eps']
    alpha = config['diffusion']['nu']

    (i,j) = fluidState['fluidNeighborhood']['indices']
    h_ij = fluidState['fluidNeighborhood']['supports']
    v_ij = fluidState['fluidVelocities'][i] - fluidState['fluidVelocities'][j]
    r_ij = fluidState['fluidNeighborhood']['distances'] * h_ij
    x_ij = fluidState['fluidNeighborhood']['vectors']# * r_ij.view(-1,1)
    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

    pi_ij = vr_ij / (r_ij + eps * h_ij**2)
    if config['diffusion']['pi-switch']:
        pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

    V_j = fluidState['fluidMasses'][j] / fluidState['fluidDensities'][j]
    kq = (V_j * pi_ij).view(-1,1) * fluidState['fluidNeighborhood']['gradients']
    return (alpha * fluidState['fluidSupports'] / config['kernel']['kernelScale'] * config['fluid']['cs'] * config['fluid']['rho0'] / fluidState['fluidDensities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = fluidState['numParticles'])


    pass
# From Cleary 1998 : Modelling confined multi-material heat and mass flows using SPH [https://doi.org/10.1016/S0307-904X(98)10031-8]
def computeViscosityCleary1998(fluidState, config):
    eps = config['diffusion']['eps']
    alpha = config['diffusion']['alpha']

    mu = 1/8 * alpha * config['fluid']['cs'] * fluidState['fluidSupports'] * fluidState['fluidDensities']

    (i,j) = fluidState['fluidNeighborhood']['indices']
    h_ij = fluidState['fluidNeighborhood']['supports']

    v_ij = fluidState['fluidVelocities'][j] - fluidState['fluidVelocities'][i]
    r_ij = fluidState['fluidNeighborhood']['distances'] * h_ij
    x_ij = fluidState['fluidNeighborhood']['vectors']# * r_ij.view(-1,1)

    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

    pi_ij = - 16 * mu[i] * mu[j] / (fluidState['fluidDensities'][i] * fluidState['fluidDensities'][j] * (mu[i] + mu[j])) * vr_ij / (r_ij + eps * h_ij**2)

    if config['diffusion']['pi-switch']:
        pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

    return scatter_sum((fluidState['fluidMasses'][j] * pi_ij).view(-1,1) * fluidState['fluidNeighborhood']['gradients'], i, dim = 0, dim_size = fluidState['numParticles'])

    pass

from diffSPH.v2.math import scatter_sum
def computeViscosityDeltaSPH_viscid(fluidState, config):
    eps = config['diffusion']['eps']
    alpha = config['diffusion']['nu']

    (i,j) = fluidState['fluidNeighborhood']['indices']
    h_ij = fluidState['fluidNeighborhood']['supports']
    v_ij = fluidState['fluidVelocities'][i] - fluidState['fluidVelocities'][j]
    r_ij = fluidState['fluidNeighborhood']['distances'] * h_ij
    x_ij = fluidState['fluidNeighborhood']['vectors']# * r_ij.view(-1,1)
    vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

    pi_ij = vr_ij / (r_ij + eps * h_ij**2)
    if config['diffusion']['pi-switch']:
        pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

    V_j = fluidState['fluidMasses'][j] / fluidState['fluidDensities'][j]
    kq = (V_j * pi_ij).view(-1,1) * fluidState['fluidNeighborhood']['gradients']
    return (2 * (config['domain']['dim'] + 2) * config['diffusion']['nu'] * config['fluid']['rho0'] / fluidState['fluidDensities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = fluidState['numParticles'])



def computeViscosity(fluidState, config):
    if config['diffusion']['velocityScheme'] == 'Monaghan1983':
        return computeViscosityMonaghan1983(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'Monaghan1992':
        return computeViscosityMonaghan1992(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'Price2012':
        return computeViscosityPrice2012(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'Cleary1998':
        return computeViscosityCleary1998(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'deltaSPH_viscid':
        return computeViscosityDeltaSPH_viscid(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'deltaSPH_inviscid':
        return computeViscosityDeltaSPH_inviscid(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'XSPH':
        return computeViscosityXSPH(fluidState, config)
    elif config['diffusion']['velocityScheme'] == 'naive':
        return computeViscosityNaive(fluidState, config)
    else:
        raise ValueError('Unknown velocity diffusion scheme')

from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('diffusion', 'velocityScheme', str, 'deltaSPH_inviscid', required = False,export = False, hint = 'Scheme for the velocity diffusion term'),
        Parameter('diffusion', 'nu', float, 0.01, required = False,export = False, hint = 'Viscosity coefficient'),
        Parameter('diffusion', 'alpha', float, 0.01, required = False,export = False, hint = 'Viscosity coefficient'),
        Parameter('diffusion', 'beta', float, 0.0, required = False,export = False, hint = 'Viscosity coefficient, used for high mach number shocks, should be 0 for low mach number flows'),
        Parameter('diffusion', 'pi-switch', bool, False, required = False,export = False, hint = 'Switches velocity diffusion off for separating particles'),
        Parameter('diffusion', 'artificial-constant', float, 0.02, required = False,export = False, hint = 'Artificial viscosity constant for XSPH'),
        Parameter('diffusion', 'eps', float, 1e-6, required = False,export = False, hint = 'Epsilon value for the viscosity term'),

    ]