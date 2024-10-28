
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
def computeViscosityMonaghan1983_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Boundary Viscosity [Monaghan 1983]"):
        eps = config['boundaryDiffusion']['eps']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        c_ij = config['fluid']['cs']
        rho_ij = (stateA['densities'][i] + stateB['densities'][j]) / 2
        alpha = config['boundaryDiffusion']['alpha']

        nu = alpha * c_ij * h_ij / rho_ij

        v_ij = stateB['velocities'][j] - stateA['velocities'][i]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)

        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = -nu * vr_ij / (r_ij + eps * h_ij**2)
        if config['boundaryDiffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

        return scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])

def computeViscosityMonaghan1992_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Monaghan 1992]"):
        eps = config['boundaryDiffusion']['eps']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateB['velocities'][j] - stateA['velocities'][i]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)

        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        c_ij = config['fluid']['cs']
        rho_ij = (stateA['densities'][i] + stateB['densities'][j]) / 2
        alpha = config['boundaryDiffusion']['alpha']
        beta = config['boundaryDiffusion']['beta']

        nu = h_ij / rho_ij * (alpha * c_ij - beta *(h_ij * vr_ij / (r_ij + eps * h_ij**2)))

        if config['boundaryDiffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

        pi_ij = -nu * vr_ij / (r_ij + eps * h_ij**2)
        return scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)


    pass
def computeViscosityXSPH_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [XSPH]"):
        alpha = config['boundaryDiffusion']['artificial-constant']

        (i,j) = neighborhood['indices']
        return alpha * sphOperationStates(stateA, stateB, stateB['velocities'][j] - stateA['velocities'][i], 'interpolate', neighborhood = neighborhood) / config['timestep']['dt']
        pass

def computeViscosityNaive_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [naive]"):

        alpha = config['boundaryDiffusion']['nu']
        return alpha * sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), 'laplacian', 'conserving', neighborhood = neighborhood) 

# From Price 2012: Smoothed particle hydrodynamics and magnetohydrodynamics [https://doi.org/10.1016/j.jcp.2010.12.011]
def computeViscosityPrice2012_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Price 2012]"):
        eps = config['boundaryDiffusion']['eps']
        alpha = config['boundaryDiffusion']['alpha']
        beta = config['boundaryDiffusion']['beta']  

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

        if config['boundaryDiffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        # print(mu_ij, pi_ij)
        return -scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])
    
from diffSPH.v2.compiler import compileSourceFiles

# sphOperation_cpp = compileSourceFiles(
    # ['/home/winchenbach/dev/diffSPH/partiBench/viscosityKernel.cpp', '/home/winchenbach/dev/diffSPH/partiBench/viscosityKernel.cu'], module_name = 'viscosityOperations', verbose = False, openMP = True, verboseCuda = False, cuda_arch = None)
# from torch.utils.cpp_extension import load

# viscosityKernel_cpp = sphOperation_cpp.viscosityKernel

def computeViscosityDeltaSPH_inviscid_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [deltaSPH inviscid]"):
        # return viscosityKernel_cpp(
        #     neighborhood['indices'], neighborhood['supports'], neighborhood['kernels'], neighborhood['gradients'], neighborhood['distances'], neighborhood['vectors'], 
        #     stateA['numParticles'],
        #     neighborhood['numNeighbors'],
        #     neighborhood['neighborOffsets'],


        #     (stateA['velocities'], stateB['velocities']), 
        #     (stateA['densities'], stateB['densities']), 
        #     (stateA['masses'], stateB['masses']), 
        #     (stateA['supports'], stateB['supports']), 
        #     config['fluid']['cs'], config['fluid']['rho0'], config['boundaryDiffusion']['alpha'], config['boundaryDiffusion']['eps'], config['boundaryDiffusion']['pi-switch'])

        eps = config['boundaryDiffusion']['eps']
        alpha = config['boundaryDiffusion']['alpha']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateA['velocities'][i] - stateB['velocities'][j]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = vr_ij / (r_ij + eps * h_ij**2)
        if config['boundaryDiffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        V_j = stateB['masses'][j] /( stateA['densities'][i] +  stateB['densities'][j])
        # if 'normals' in stateB:
        #     boundaryNormals = stateB['normals'][j]
        #     v_ij_parallel = torch.einsum('ij,ij->i', v_ij, boundaryNormals).view(-1,1) * boundaryNormals
        #     v_ij_orthogonal = v_ij - v_ij_parallel

        #     vr_ij = torch.einsum('ij,ij->i', vr_ij, x_ij)

        #     pi_ij = vr_ij / (r_ij + eps * h_ij**2)
        #     if config['boundaryDiffusion']['pi-switch']:
        #         pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        kq = (V_j * pi_ij).view(-1,1) * neighborhood['gradients']
        viscosityTerm =  (alpha * stateA['supports'] * config['fluid']['cs'] * config['fluid']['rho0'] / stateA['densities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = stateA['numParticles'])
    


        return viscosityTerm
        # return (alpha * fluidState['fluidSupports'] * config['fluid']['cs'] * config['fluid']['rho0'] / fluidState['densities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = fluidState['numParticles'])


# From Cleary 1998 : Modelling confined multi-material heat and mass flows using SPH [https://doi.org/10.1016/S0307-904X(98)10031-8]
def computeViscosityCleary1998_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [Cleary 1998]"):
        eps = config['boundaryDiffusion']['eps']
        alpha = config['boundaryDiffusion']['alpha']

        muA = 1/8 * alpha * config['fluid']['cs'] * stateA['supports'] * stateA['densities']
        muB = 1/8 * alpha * config['fluid']['cs'] * stateB['supports'] * stateB['densities']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']

        v_ij = stateB['velocities'][j] - stateA['velocities'][i]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)

        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = - 16 * muA[i] * muB[j] / (stateA['densities'][i] * stateB['densities'][j] * (muA[i] + muB[j])) * vr_ij / (r_ij + eps * h_ij**2)

        if config['boundaryDiffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij > 0, pi_ij, 0)

        return scatter_sum((stateB['masses'][j] * pi_ij).view(-1,1) * neighborhood['gradients'], i, dim = 0, dim_size = stateA['numParticles'])

from diffSPH.v2.math import scatter_sum
def computeViscosityDeltaSPH_viscid_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity [deltaSPH viscid]"):
        eps = config['boundaryDiffusion']['eps']
        alpha = config['boundaryDiffusion']['nu']

        (i,j) = neighborhood['indices']
        h_ij = neighborhood['supports']
        v_ij = stateA['velocities'][i] - stateB['velocities'][j]
        r_ij = neighborhood['distances'] * h_ij
        x_ij = neighborhood['vectors']# * r_ij.view(-1,1)
        vr_ij = torch.einsum('ij,ij->i', v_ij, x_ij)

        pi_ij = vr_ij / (r_ij + eps * h_ij**2)
        if config['boundaryDiffusion']['pi-switch']:
            pi_ij = torch.where(vr_ij < 0, pi_ij, 0)

        V_j = stateB['masses'][j] / stateB['densities'][j]
        kq = (V_j * pi_ij).view(-1,1) * neighborhood['gradients']
        return (2 * (config['domain']['dim'] + 2) * config['boundaryDiffusion']['nu'] * config['fluid']['rho0'] / stateA['densities']).view(-1,1) * scatter_sum(kq, i, dim = 0, dim_size = stateA['numParticles'])



def computeViscosity_Boundary(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Viscosity (nu Delta v)"):
        if config['boundaryDiffusion']['velocityScheme'] == 'Monaghan1983':
            return computeViscosityMonaghan1983_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'Monaghan1992':
            return computeViscosityMonaghan1992_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'Price2012':
            return computeViscosityPrice2012_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'Cleary1998':
            return computeViscosityCleary1998_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'deltaSPH_viscid':
            return computeViscosityDeltaSPH_viscid_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'deltaSPH_inviscid':
            return computeViscosityDeltaSPH_inviscid_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'XSPH':
            return computeViscosityXSPH_Boundary(stateA, stateB, neighborhood, config)
        elif config['boundaryDiffusion']['velocityScheme'] == 'naive':
            return computeViscosityNaive_Boundary(stateA, stateB, neighborhood, config)
        else:
            raise ValueError('Unknown velocity diffusion scheme')

from diffSPH.parameter import Parameter


def getParameters():
    return [
        Parameter('boundaryDiffusion', 'velocityScheme', str, None, required = False,export = False, hint = 'Scheme for the velocity diffusion term'),
        Parameter('boundaryDiffusion', 'nu', float, None, required = False,export = False, hint = 'Viscosity coefficient'),
        Parameter('boundaryDiffusion', 'alpha', float, None, required = False,export = False, hint = 'Viscosity coefficient'),
        Parameter('boundaryDiffusion', 'beta', float, None, required = False,export = False, hint = 'Viscosity coefficient, used for high mach number shocks, should be 0 for low mach number flows'),
        Parameter('boundaryDiffusion', 'pi-switch', bool, None, required = False,export = False, hint = 'Switches velocity diffusion off for separating particles'),
        Parameter('boundaryDiffusion', 'artificial-constant', float, None, required = False,export = False, hint = 'Artificial viscosity constant for XSPH'),
        Parameter('boundaryDiffusion', 'eps', float, None, required = False,export = False, hint = 'Epsilon value for the viscosity term'),

    ]



from diffSPH.v2.sphOps import sphOperationStates, LiuLiuConsistent
from diffSPH.v2.util import countUniqueEntries

def computeBoundaryVelocities(perennialState, config):
    ghostState = perennialState['boundaryGhost']
    ghostNormal = -2 * perennialState['boundary']['distances'].view(-1,1) * perennialState['boundary']['normals']

    ids = perennialState['boundary']['bodyIDs']
    regions = [region for region in config['regions'] if region['type'] == 'boundary']

    uniqueIDs,_ = countUniqueEntries(perennialState['boundary']['bodyIDs'].to(torch.int32), perennialState['boundary']['positions'])

    velocities = torch.zeros_like(perennialState['boundary']['velocities'])

    for ui, u in enumerate(uniqueIDs):
        mode = regions[ui]['kind']
        # print(ui, mode, u)
        # print(ids == u)

        if mode == 'linear':
            solution_ux, M_ux, b_ux = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,0])
            projected_ux = solution_ux[:,0] + torch.einsum('nd, nd -> n', ghostNormal, solution_ux[:,1:])
            solution_uy, M_uy, b_uy = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,1])
            projected_uy = solution_uy[:,0] + torch.einsum('nd, nd -> n', ghostNormal, solution_uy[:,1:])

            velocities[ids == u,:] = torch.stack((projected_ux, projected_uy), dim = -1)[ids == u,:]
        elif mode == 'zero':
            velocities[ids == u,:] = torch.zeros_like(perennialState['boundary']['velocities'])[ids == u,:]
        elif mode == 'none':
            velocities[ids == u,:] = perennialState['boundary']['velocities'].clone()[ids == u,:]
        elif mode == 'free-slip':
            solution_ux, M_ux, b_ux = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,0])
            projected_ux = solution_ux[:,0] #+ torch.einsum('nd, nd -> n', ghostNormal, solution_ux[:,1:])
            solution_uy, M_uy, b_uy = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,1])
            projected_uy = solution_uy[:,0] #+ torch.einsum('nd, nd -> n', ghostNormal, solution_uy[:,1:])
            cvelocities = torch.stack((projected_ux, projected_uy), dim = -1)

            velocities[ids == u,:] = (cvelocities - torch.einsum('nd, nd -> n', cvelocities, perennialState['boundary']['normals']).view(-1,1) * perennialState['boundary']['normals'])[ids == u,:]
        elif mode == 'no-slip':
            solution_ux, M_ux, b_ux = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,0])
            projected_ux = solution_ux[:,0] #+ torch.einsum('nd, nd -> n', ghostNormal, solution_ux[:,1:])
            solution_uy, M_uy, b_uy = LiuLiuConsistent(ghostState, perennialState['fluid'], perennialState['fluid']['velocities'][:,1])
            projected_uy = solution_uy[:,0] #+ torch.einsum('nd, nd -> n', ghostNormal, solution_uy[:,1:])
            velocities = torch.stack((projected_ux, projected_uy), dim = -1)

            velocities[ids == u,:] = (cvelocities - torch.einsum('nd, nd -> n', cvelocities, perennialState['boundary']['normals']).view(-1,1) * perennialState['boundary']['normals'])[ids == u,:]
            # velocities = - velocities
        else:
            raise ValueError(f'Unknown boundary condition {mode}')

    return velocities