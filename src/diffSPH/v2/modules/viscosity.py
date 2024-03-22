
import torch
import torch
from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.math import mod, scatter_sum
import numpy as np
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.math import scatter_sum

from torch.profiler import record_function

# def computeViscosityMonaghan(simulationState, config):
#     with record_function("Viscosity [Monaghan]"):       
#         return config['fluid']['mu'] * sphOperationStates(stateA, stateB, (simulationState['velocities'], simulationState['velocities']), operation = 'laplacian', gradientMode='conserving')# / simulationState['densities'].view(-1,1)

# simulationState['viscosity'] = physicalParameters['nu'] * sphOperation(
#     (areas, areas), 
#     (simulationState['rho'], simulationState['rho']),
#     (u, u),
#     (i, j), kernel, gradKernel, rij, xij, hij, x.shape[0], operation = 'laplacian', gradientMode = 'conserving') / simulationState['rho'].view(-1,1)

from diffSPH.v2.math import scatter_sum
def computeViscosityMonaghan1983(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Monaghan 1983]"):
        eps = config['diffusion']['eps']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        c_ij = config['fluid']['cs']
        rho_ij = (stateA['densities'][i] + stateB['densities'][j]) / 2
        alpha = config['diffusion']['alpha']

        nu = alpha * c_ij * h_ij / rho_ij

        v_ij = stateB['velocities'][j] - stateA['velocities'][i]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)

        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = -nu * vr_ij / (r_ij + eps * h_ij**2)
        if config['diffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

        return scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])

def computeViscosityMonaghan1992(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Monaghan 1992]"):
        eps = config['diffusion']['eps']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateB['velocities'][j] - stateA['velocities'][i]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)

        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        c_ij = config['fluid']['cs']
        rho_ij = (stateA['densities'][i] + stateB['densities'][j]) / 2
        alpha = config['diffusion']['alpha']
        beta = config['diffusion']['beta']

        nu = h_ij / rho_ij * (alpha * c_ij - beta *(h_ij * vr_ij / (r_ij + eps * h_ij**2)))

        if config['diffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

        pi_ij = -nu * vr_ij / (r_ij + eps * h_ij**2)
        return scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)


    pass
def computeViscosityXSPH(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [XSPH]"):
        alpha = config['diffusion']['artificial-constant']

        (i,j) = neighborhood['indices']
        return alpha * sphOperationStates(stateA, stateB, stateB['velocities'][j] - stateA['velocities'][i], 'interpolate', neighborhood = neighborhood) / config['timestep']['dt']
        pass

def computeViscosityNaive(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [naive]"):

        alpha = config['diffusion']['nu']
        return alpha * sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), 'laplacian', 'conserving', neighborhood = neighborhood) 

# From Price 2012: Smoothed particle hydrodynamics and magnetohydrodynamics [https://doi.org/10.1016/j.jcp.2010.12.011]
def computeViscosityPrice2012(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Price 2012]"):
        eps = config['diffusion']['eps']
        alpha = config['diffusion']['alpha']
        beta = config['diffusion']['beta']  

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateA['velocities'][i] - stateB['velocities'][j]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        mu_ij = h_ij * vr_ij / (r_ij + eps * h_ij**2)

        c_ij = config['fluid']['cs']
        rho_ij = (stateA['densities'][i] + stateB['densities'][j]) / 2
        pi_ij = (-alpha * c_ij * mu_ij + beta  * mu_ij**2 ) / rho_ij

        if config['diffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        # print(mu_ij, pi_ij)
        return -scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])
    
def computeViscosityDeltaSPH_inviscid(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [deltaSPH inviscid]"):
        eps = config['diffusion']['eps']
        alpha = config['diffusion']['alpha']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateA['velocities'][i] - stateB['velocities'][j]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = vr_ij / (r_ij + eps * h_ij**2)
        if config['diffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        V_j = stateB['masses'][j] / stateA['densities'][j]
        kq = (V_j * pi_ij).view(-1,1) * neighborhood['gradients']
        return (alpha * stateA['supports'] * config['fluid']['cs'] * config['fluid']['rho0'] / stateA['densities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = stateA['numParticles'])
        # return (alpha * fluidState['fluidSupports'] * config['fluid']['cs'] * config['fluid']['rho0'] / fluidState['densities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = fluidState['numParticles'])


# From Cleary 1998 : Modelling confined multi-material heat and mass flows using SPH [https://doi.org/10.1016/S0307-904X(98)10031-8]
def computeViscosityCleary1998(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Cleary 1998]"):
        eps = config['diffusion']['eps']
        alpha = config['diffusion']['alpha']

        muA = 1/8 * alpha * config['fluid']['cs'] * stateA['supports'] * stateA['densities']
        muB = 1/8 * alpha * config['fluid']['cs'] * stateB['supports'] * stateB['densities']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']

        v_ij = stateB['velocities'][j] - stateA['velocities'][i]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)

        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = - 16 * muA[i] * muB[j] / (stateA['densities'][i] * stateB['densities'][j] * (muA[i] + muB[j])) * vr_ij / (r_ij + eps * h_ij**2)

        if config['diffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

        return scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])

from diffSPH.v2.math import scatter_sum
def computeViscosityDeltaSPH_viscid(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [deltaSPH viscid]"):
        eps = config['diffusion']['eps']
        alpha = config['diffusion']['nu']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateA['velocities'][i] - stateB['velocities'][j]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = vr_ij / (r_ij + eps * h_ij**2)
        if config['diffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        V_j = stateB['masses'][j] / stateB['densities'][j]
        kq = (V_j * pi_ij).view(-1,1) * neighborhood['gradients']
        return (2 * (config['domain']['dim'] + 2) * config['diffusion']['nu'] * config['fluid']['rho0'] / stateA['densities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = stateA['numParticles'])



def computeViscosity(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity (nu Delta v)"):
        if config['diffusion']['velocityScheme'] == 'Monaghan1983':
            return computeViscosityMonaghan1983(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'Monaghan1992':
            return computeViscosityMonaghan1992(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'Price2012':
            return computeViscosityPrice2012(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'Cleary1998':
            return computeViscosityCleary1998(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'deltaSPH_viscid':
            return computeViscosityDeltaSPH_viscid(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'deltaSPH_inviscid':
            return computeViscosityDeltaSPH_inviscid(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'XSPH':
            return computeViscosityXSPH(stateA, stateB, neighborhood, config)
        elif config['diffusion']['velocityScheme'] == 'naive':
            return computeViscosityNaive(stateA, stateB, neighborhood, config)
        else:
            raise ValueError('Unknown velocity diffusion scheme')

from diffSPH.parameter import Parameter

def computeViscosityParameter(particleState, config):      
    nu_sph = config['diffusion']['alpha'] * config['fluid']['cs'] * config['particle']['support']  / (2 * (config['domain']['dim'] + 2)) * 5/4
    nu_sph = config['diffusion']['nu'] if config['diffusion']['velocityScheme'] == 'deltaSPH_viscid' else nu_sph
    return nu_sph



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