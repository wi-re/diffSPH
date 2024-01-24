import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
from torch.profiler import profile, record_function, ProfilerActivity

from ..kernels import kernel, kernelGradient
from ..module import Module, BoundaryModule
from ..parameter import Parameter
from ..util import *

from ..ghostParticles import *

from .deltaSPH import *
# from .diffusion import computeVelocityDiffusion
from .densityDiffusion import *
# from .momentum import computeDivergenceTerm
# from .pressure import computePressureAccel

from .densityDiffusion import *
from ..kernels import kernel, kernelGradient, spikyGrad, wendland, wendlandGrad, cohesionKernel, getKernelFunctions
from .neighborSearch import periodicNeighborSearch


def neighborhood(positions, h, minDomain, maxDomain, periodicX, periodicY):
    i, j, fluidDistances, fluidRadialDistances = periodicNeighborSearch(positions, minDomain, maxDomain, h, periodicX, periodicY, useCompactHashMap = True)

    # j, i = radius(positions, positions, h,max_num_neighbors = 1000)
    cartesianDistances = positions[j] - positions[i]
    cartesianDistances[:,0] = torch.remainder(cartesianDistances[:,0] + minDomain[0], maxDomain[0] - minDomain[0]) - maxDomain[0]
    cartesianDistances[:,1] = torch.remainder(cartesianDistances[:,1] + minDomain[1], maxDomain[1] - minDomain[1]) - maxDomain[1]
    radialDistances = torch.linalg.norm(cartesianDistances, dim = 1) / h
    cDistances = cartesianDistances / ((radialDistances * h)[:,None] + 0.0001 * h**2)
    
    return i, j, radialDistances, cDistances, cartesianDistances 

@torch.jit.script
def pinv2x2(M):
    with record_function('Pseudo Inverse 2x2'):
        a = M[:,0,0]
        b = M[:,0,1]
        c = M[:,1,0]
        d = M[:,1,1]

        theta = 0.5 * torch.atan2(2 * a * c + 2 * b * d, a**2 + b**2 - c**2 - d**2)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        U = torch.zeros_like(M)
        U[:,0,0] = cosTheta
        U[:,0,1] = - sinTheta
        U[:,1,0] = sinTheta
        U[:,1,1] = cosTheta

        S1 = a**2 + b**2 + c**2 + d**2
        S2 = torch.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4* (a * c + b *d)**2)

        o1 = torch.sqrt((S1 + S2) / 2)
        o2 = torch.sqrt((S1 - S2) / 2)

        phi = 0.5 * torch.atan2(2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2)
        cosPhi = torch.cos(phi)
        sinPhi = torch.sin(phi)
        s11 = torch.sign((a * cosTheta + c * sinTheta) * cosPhi + ( b * cosTheta + d * sinTheta) * sinPhi)
        s22 = torch.sign((a * sinTheta - c * cosTheta) * sinPhi + (-b * sinTheta + d * cosTheta) * cosPhi)

        V = torch.zeros_like(M)
        V[:,0,0] = cosPhi * s11
        V[:,0,1] = - sinPhi * s22
        V[:,1,0] = sinPhi * s11
        V[:,1,1] = cosPhi * s22


        o1_1 = torch.zeros_like(o1)
        o2_1 = torch.zeros_like(o2)

        o1_1[torch.abs(o1) > 1e-5] = 1 / o1[torch.abs(o1) > 1e-5] 
        o2_1[torch.abs(o2) > 1e-5] = 1 / o2[torch.abs(o2) > 1e-5] 
        o = torch.vstack((o1_1, o2_1))
        S_1 = torch.diag_embed(o.mT, dim1 = 2, dim2 = 1)

        eigVals = torch.vstack((o1, o2)).mT
        eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])

        return torch.matmul(torch.matmul(V, S_1), U.mT), eigVals

    
@torch.jit.script
def computeNormalizationMatrix(i, j, rij, Vi, Vj, distances, radialDistances, support : float, numParticles : int, eps : float):
    gradW = kernelGradient(radialDistances, distances, support)

    r_ba = -rij #* support / 2
    fac = Vj[j]

#     term = torch.einsum('nu,nv -> nuv', r_ba, gradW)
#     term[:,0,0] = term[:,0,0] * fac
#     term[:,0,1] = term[:,0,1] * fac
#     term[:,1,0] = term[:,1,0] * fac
#     term[:,1,1] = term[:,1,1] * fac
    term = torch.zeros((r_ba.shape[0],2,2)).to(r_ba.device).type(r_ba.dtype)
    term[:,0,0] = gradW[:,0] * r_ba[:,0] * fac
    term[:,0,1] = gradW[:,0] * r_ba[:,1] * fac
    term[:,1,0] = gradW[:,1] * r_ba[:,0] * fac
    term[:,1,1] = gradW[:,1] * r_ba[:,1] * fac

    fluidNormalizationMatrix = scatter(term, i, dim=0, dim_size=numParticles, reduce = 'sum')
    return fluidNormalizationMatrix
@torch.jit.script
def computeTerm(i, j, Vi, Vj, distances, radialDistances, support : float, numParticles : int, eps : float):
    gradW = kernelGradient(radialDistances, distances, support)

    fac = Vj[j]

    fluidNormalizationMatrix = scatter(fac[:,None] * gradW, i, dim=0, dim_size=numParticles, reduce = 'sum')
    return fluidNormalizationMatrix

# sqrt2 = np.sqrt(2)
from typing import List
@torch.jit.script
def computeFreeSurface(i, j, positions, L, lambdas, volume, rij, cartesianDistances, radialDistances, h : float, numParticles : int, minDomain: List[float], maxDomain: List[float]):
    term = computeTerm(i, j, volume, volume, cartesianDistances, radialDistances, h, numParticles, 1e-4)

    nu = torch.bmm(L, term.unsqueeze(2))[:,:,0]
    n = nu / (torch.linalg.norm(nu, dim = 1) + 0.01 * h**2)[:,None]
    lMin = torch.min(torch.abs(lambdas), dim = 1)[0]
# #   Plot particle normals  
# #     fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
# #     mask = torch.logical_and(lMin > 0.25, lMin < 0.75)
# #     scatterPlot(fig, axis[0,0], positions[mask], n[mask,0], 'n.x')
# #     scatterPlot(fig, axis[0,1], positions[mask], n[mask,1], 'n.y')
# #     axis[0,2].quiver(positions[mask,0], positions[mask,1], n[mask,0], n[mask,1])
# #     axis[0,2].axis('equal')
# #     fig.tight_layout()

    T = positions + n * h / 3

    tau = torch.vstack((-n[:,1], n[:,0])).mT
    xji = -rij
    xjt = positions[j] - T[i]
    # xjt = torch.remainder(xjt + minDomain, maxDomain - minDomain) - maxDomain
    xjt[:,0] = torch.remainder(xjt[:,0] + minDomain[0], maxDomain[0] - minDomain[0]) - maxDomain[0]
    xjt[:,1] = torch.remainder(xjt[:,1] + minDomain[1], maxDomain[1] - minDomain[1]) - maxDomain[1]

    condA1 = torch.linalg.norm(xji, dim = 1) >= torch.sqrt(torch.tensor(2)) * h /3
    condA2 = torch.linalg.norm(xjt, dim = 1) < h / 3
    condA = torch.logical_and(torch.logical_and(condA1, condA2), i != j)
    cA = scatter(condA, i, dim=0, dim_size = numParticles, reduce = 'sum')
    condB1 = torch.linalg.norm(xji, dim = 1) < torch.sqrt(torch.tensor(2)) * h /3
    condB2 =  torch.abs(torch.einsum('nu,nu -> n', -n[i], xjt)) + torch.abs(torch.einsum('nu,nu -> n', tau[i], xjt)) < h / 3
    condB =  torch.logical_and(torch.logical_and(condB1, condB2), i != j)
    cB = scatter(condB, i, dim=0, dim_size = numParticles, reduce = 'sum')
    fs = torch.where(torch.logical_and(torch.logical_not(cA), torch.logical_not(cB)), torch.ones_like(lMin), torch.zeros_like(lMin))
#   Plot FS Conditions  
#     print('condition A1, true for: ', torch.sum(condA1), 'false for: ', torch.sum(torch.logical_not(condA1)), 'rate: %4.4g %%'% ((torch.sum(condA1) / condA1.shape[0]).numpy() * 100))
#     print('condition A2, true for: ', torch.sum(condA2), 'false for: ', torch.sum(torch.logical_not(condA2)), 'rate: %4.4g %%'% ((torch.sum(condA2) / condA2.shape[0]).numpy() * 100))
#     print('condition A, true for: ', torch.sum(condA), 'false for: ', torch.sum(torch.logical_not(condA)), 'rate: %4.4g %%'% ((torch.sum(condA) / condA.shape[0]).numpy() * 100))


#     print('condition B1, true for: ', torch.sum(condB1), 'false for: ', torch.sum(torch.logical_not(condB1)), 'rate: %4.4g %%'% ((torch.sum(condB1) / condB1.shape[0]).numpy() * 100))
#     print('condition B2, true for: ', torch.sum(condB2), 'false for: ', torch.sum(torch.logical_not(condB2)), 'rate: %4.4g %%'% ((torch.sum(condB2) / condB2.shape[0]).numpy() * 100))
#     print('condition B, true for: ', torch.sum(condB), 'false for: ', torch.sum(torch.logical_not(condB)), 'rate: %4.4g %%'% ((torch.sum(condB) / condB.shape[0]).numpy() * 100))

#     cA1 = scatter(condA1, i, dim=0, dim_size = numParticles, reduce = 'sum')
#     cA2 = scatter(condA2, i, dim=0, dim_size = numParticles, reduce = 'sum')
#     cB1 = scatter(condB1, i, dim=0, dim_size = numParticles, reduce = 'sum')
#     cB2 = scatter(condB2, i, dim=0, dim_size = numParticles, reduce = 'sum')

#     # fig, axis = plt.subplots(2, 3, figsize=(18,12), sharex = False, sharey = False, squeeze = False)
#     # scatterPlot(fig, axis[0,0], positions, torch.where(cA1, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'condA1')
#     # scatterPlot(fig, axis[0,1], positions, torch.where(cA2, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'condA2')
#     # scatterPlot(fig, axis[0,2], positions, torch.where(cA, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'condA')
#     # scatterPlot(fig, axis[1,0], positions, torch.where(cB1, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'condB1')
#     # scatterPlot(fig, axis[1,1], positions, torch.where(cB2, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'condB2')
#     # scatterPlot(fig, axis[1,2], positions, torch.where(cB, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'condB')

#     fig, axis = plt.subplots(1, 1, figsize=(6,6), sharex = False, sharey = False, squeeze = False)
#     scatterPlot(fig, axis[0,0], positions, torch.where(fs, torch.ones(positions.shape[0]), torch.zeros(positions.shape[0])), 'FS')
#     fig.tight_layout()
    
    return fs

@torch.jit.script
def computeShiftingTerm(i, j, Vi, Vj, distances, radialDistances, corr : float, support : float, numParticles : int, eps : float):
    R = 0.2
    n = 4
    W = kernel(radialDistances, support) / corr
    term = 1. + R * torch.pow(W, n)
    
    gradW = kernelGradient(radialDistances, distances, support)
    
    s = (Vj[j] * term)[:,None] * gradW
    fluidNormalizationMatrix = scatter(s, i, dim=0, dim_size=numParticles, reduce = 'sum')
    return fluidNormalizationMatrix

@torch.jit.script
def computeNormals(i, j, Vi, Vj, li, lj, Li, Lj, distances, radialDistances, support : float, numParticles : int, eps : float):
    
    gradW = kernelGradient(radialDistances, distances, support)
    kernelTerm = torch.bmm(Li[i], gradW.unsqueeze(2))[:,:,0]
    
    fac = Vj[j] * (lj[j] - li[i])
    term = -fac[:,None] * kernelTerm
    fluidNormalizationMatrix = scatter(term, i, dim=0, dim_size=numParticles, reduce = 'sum')
    return fluidNormalizationMatrix

@torch.jit.script
def computeShiftAmount(i, j, volume, lambdas, L, expandedFSM, cartesianDistances, radialDistances, h : float, numParticles : int, Ma : float, c0 : float, dx : float):
    lMin = torch.min(torch.abs(lambdas), dim = 1)[0]
    gradLambda = computeNormals(i, j, volume, volume, lMin, lMin, L, L, cartesianDistances, radialDistances, h, numParticles, 1e-4)
    normals = gradLambda / (torch.linalg.norm(gradLambda, dim = 1) + 1e-4 * h**2)[:,None]

    normals[expandedFSM < 1,:] = 0
    kappa = torch.where(scatter(torch.arccos((normals[i] * normals[j]).sum(1)), i, dim = 0, dim_size = numParticles, reduce = 'max') >= 15 * np.pi/180, torch.zeros(numParticles), torch.ones(numParticles))
    fac = - Ma * (2 * h / 3) * c0 
#     Plot ev normals
#     mask = expandedFSM > 0

#     fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
#     scatterPlot(fig, axis[0,0], positions[mask], normals[mask,0] , 'dr.x')
#     scatterPlot(fig, axis[0,1], positions[mask], normals[mask,1] , 'dr.y')
#     scatterPlot(fig, axis[0,2], positions[mask], kappa[mask] , 'kappa')
#     fig.tight_layout()
    du = -fac * computeShiftingTerm(i, j, volume, volume, cartesianDistances, radialDistances, kernelScalar(dx / h, h), h, numParticles, 1e-4)

    shiftTermA = torch.zeros_like(du)
    shiftTermB = kappa[:,None] * torch.bmm((torch.eye(2).reshape((1,2,2)).repeat(numParticles,1,1).to(normals.device) - torch.einsum('nu, nv -> nuv', normals, normals)), du.unsqueeze(2))[:,:,0]
    shiftTermB = torch.bmm((torch.eye(2).reshape((1,2,2)).repeat(numParticles,1,1).to(normals.device) - torch.einsum('nu, nv -> nuv', normals, normals)), du.unsqueeze(2))[:,:,0]
    shiftTermC = du

    prodTerm = -(normals * du).sum(1)
#     print(prodTerm[torch.logical_and(lMin >= 0.55, expandedFSM > 0)])
    lambdaThreshold = 0.55
    condA = torch.logical_and(lMin >= lambdaThreshold, torch.logical_and(prodTerm >= 0, expandedFSM > 0))
    condB = torch.logical_and(lMin >= lambdaThreshold, torch.logical_and(prodTerm < 0, expandedFSM > 0))
    condC = expandedFSM < 1
    
#     print(torch.max(prodTerm))

    shiftVelocity = shiftTermA
    shiftVelocity[condA,:] = shiftTermB[condA,:]
    shiftVelocity[condB,:] = du[condB,:]
    shiftVelocity[condC,:] = du[condC,:]
    
    return shiftVelocity, normals#, condA, condB, condC, kappa

class deltaPlusModule(Module):
#     def getParameters(self):
#         return [
#             # Parameter('deltaSPH', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'delta', 'float', 0.1, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'c0', 'float', -1.0, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'beta', 'float', -1.0, required = False, export = True, hint = ''),
#             Parameter('deltaSPH', 'HughesGrahamCorrection', 'bool', True, required = False, export = True, hint = '')
#         ]
        
    def exportState(self, simulationState, simulation, grp, mask):  
        if simulation.config['shifting']['enabled']:
            grp.create_dataset('fluidShifting', data = simulationState['fluidUpdate'].detach().cpu().numpy())


    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        self.restDensity = simulationConfig['fluid']['restDensity']
        self.gamma = simulationConfig['pressure']['gamma']
        
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.pressureScheme = simulationConfig['pressure']['fluidPressureTerm'] 
        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device'] 
        
        self.alpha = simulationConfig['diffusion']['alpha']
        self.delta = simulationConfig['diffusion']['delta'] 
        self.dx = simulationConfig['particle']['support'] * simulationConfig['particle']['packing']
        c0 = 10.0 * np.sqrt(2.0*9.81*0.3)
        h0 = simulationConfig['particle']['support']
        dt = 0.25 * h0 / (1.1 * c0)
        if simulationConfig['fluid']['c0'] < 0:
            simulationConfig['fluid']['c0'] = c0
        
        self.c0 = simulationConfig['fluid']['c0']
        self.eps = self.support **2 * 0.1
        self.scheme = simulationConfig['diffusion']['densityScheme']
        self.umax = simulationConfig['fluid']['umax']

        self.support = np.sqrt(50 / np.pi * simulationConfig['particle']['area'])

        self.minDomain = simulationConfig['domain']['min']
        self.maxDomain = simulationConfig['domain']['max']
        self.periodicX = simulationConfig['periodicBC']['periodicX']
        self.periodicY = simulationConfig['periodicBC']['periodicY']
        # self.dx = simulationConfig['particle']['packing'] *
        

    def computeShiftStep(self, simulationState, simulation):
    #     numParticles = int(numParticles)
        h = self.support
        positions = simulationState['fluidPosition']
        numParticles = positions.shape[0]
        area = simulationState['fluidArea']

        i, j, radialDistances, cartesianDistances, rij = neighborhood(positions, h, self.minDomain, self.maxDomain, self.periodicX, self.periodicY)
        # rho = scatter(area * kernel(radialDistances, h), i, dim = 0, dim_size = numParticles, reduce = 'sum')
        # neighs = scatter(torch.ones_like(radialDistances), i, dim = 0, dim_size = numParticles, reduce = 'sum')
        volume = area / simulationState['fluidDensity']
        normalizationMatrices = computeNormalizationMatrix(i, j, rij, volume, volume, cartesianDistances, radialDistances, h, numParticles, 1e-4)

        L, lambdas = pinv2x2(normalizationMatrices)
        # Linv, invLambdas = pinv2x2(L)
        # invLambdas = 1 / lambdas
        fs = computeFreeSurface(i,j, positions, L, lambdas, volume, rij, cartesianDistances, radialDistances, h, numParticles, self.minDomain, self.maxDomain)
        expandedFSM = scatter(fs[j], i, dim = 0, dim_size = numParticles, reduce = 'max')
        du, normals = computeShiftAmount(i, j, volume, lambdas, L, expandedFSM, cartesianDistances, radialDistances, h, numParticles, self.umax / self.c0, simulation.config['fluid']['c0'], self.dx)
        
        dmag = torch.clamp(torch.linalg.norm(du, dim = 1), max = self.umax/2)
    #     print(torch.linalg.norm(du, dim = 1))
    #     print(dmag)
        du = (dmag / (torch.linalg.norm(du, dim = 1) + 1e-4 * h**2))[:,None] * du
    #     print(du)
        # dr = dt * du
        
        simulationState['fluidUpdate'] = du

    def computeShiftAmount(self, simulationState, simulation):
        i,j = simulationState['fluidNeighbors']
        support = self.support
        eps = support **2 * 0.1

        CFL = 1.5
        supportTerm = 4 * support**2
        Ma = torch.linalg.norm(simulationState['fluidVelocity'], dim = 1) / simulation.config['fluid']['c0']
        Ma = torch.max(torch.linalg.norm(simulationState['fluidVelocity'], dim = 1)) / simulation.config['fluid']['c0']
        k0 = kernel(simulation.config['particle']['packing'], support)
        R = 0.2
        n = 4

        kernelTerm = 1 + R * torch.pow(kernel(simulationState['fluidRadialDistances'], support) / k0, n)
        gradientTerm = kernelGradient(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], support)

        phi_ij = 1

        massTerm = simulationState['fluidArea'][j] / (simulationState['fluidDensity'][i] + simulationState['fluidDensity'][j])

        term = (kernelTerm * massTerm * phi_ij )[:,None] * gradientTerm

        simulationState['shiftAmount'] = - CFL * Ma * supportTerm * scatter_sum(term, i, dim=0, dim_size = simulationState['fluidDensity'].shape[0])
        if hasattr(self, 'boundaryModule') and simulation.boundaryModule.active:
            bb, bf = simulation.boundaryModule.boundaryToFluidNeighbors

            kernelTerm = 1 + R * torch.pow(kernel(simulation.boundaryModule.boundaryToFluidNeighborRadialDistances, support) / k0, n)
            gradientTerm = kernelGradient(simulation.boundaryModule.boundaryToFluidNeighborRadialDistances, -simulation.boundaryModule.boundaryToFluidNeighborDistances, support)

            phi_ij = 1

            massTerm = simulation.boundaryModule.boundaryVolume[bb] / (simulationState['fluidDensity'][bf] + simulation.boundaryModule.boundaryDensity[bb])
            term = (kernelTerm * massTerm * phi_ij )[:,None] * gradientTerm

            simulationState['shiftAmount'] += - CFL * Ma * supportTerm * scatter_sum(term, bf, dim=0, dim_size = simulationState['fluidDensity'].shape[0])
    def computeNormalizationmatrix(self, simulationState, simulation):
        support = self.support
        eps = support **2 * 0.1

        if 'fluidL' in simulationState:
            self.normalizationMatrix = simulationState['normalizationMatrix']
            self.fluidL = simulationState['fluidL']
            self.eigVals = simulationState['eigVals']
        else:
            simulationState['normalizationMatrix'] = computeNormalizationMatrix(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                          simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                          simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                          support, simulationState['fluidDensity'].shape[0], eps)     
            if hasattr(self, 'boundaryModule') and simulation.boundaryModule.active:
                simulationState['normalizationMatrix'] += simulation.boundaryModule.computeNormalizationMatrices(simulationState, simulation)
            simulationState['fluidL'], simulationState['eigVals'] = pinv2x2(simulationState['normalizationMatrix'])
        simulationState['fluidLambda'] = simulationState['eigVals'][:,1]
    def computeFluidNormal(self, simulationState, simulation):
        i,j = simulationState['fluidNeighbors']
        support = self.support
        eps = support **2 * 0.1

        
        volume = simulationState['fluidArea'][j] / simulationState['fluidDensity'][j]
        factor = simulationState['fluidLambda'][j] - simulationState['fluidLambda'][i]

        kernelGrad = simulation.kernelGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], support)

        correctedKernel = torch.bmm(simulationState['fluidL'][i], kernelGrad[:,:,None])
        # print(correctedKernel.shape)

        term = -(volume * factor)[:,None] * correctedKernel[:,:,0]

        simulationState['lambdaGrad'] = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
        if hasattr(self, 'boundaryModule') and simulation.boundaryModule.active:
            bb, bf = simulation.boundaryModule.boundaryToFluidNeighbors
            volume = simulationState['fluidArea'][bb] / simulationState['fluidDensity'][bb]
            factor = simulation.boundaryModule.eigVals[:,1][bb] - simulationState['fluidLambda'][bf]

            kernelGrad = kernelGradient(simulation.boundaryModule.boundaryToFluidNeighborRadialDistances, -simulation.boundaryModule.boundaryToFluidNeighborDistances, support)

            correctedKernel = torch.bmm(simulationState['fluidL'][bf], kernelGrad[:,:,None])
            # print(correctedKernel.shape)

            term = -(volume * factor)[:,None] * correctedKernel[:,:,0]

            simulationState['lambdaGrad'] += scatter(term, bf, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        simulationState['fluidNormal'] = simulationState['lambdaGrad'] / (torch.linalg.norm(simulationState['lambdaGrad'],dim=1) + eps)[:,None]
        
    def detectFluidSurface(self, simulationState, simulation):
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[0, neighbors[0] != neighbors[1]]
        j = neighbors[1, neighbors[0] != neighbors[1]]
        support = self.support
        eps = support **2 * 0.1


        gradients = simulationState['fluidNormal'][i]
        distances = -simulationState['fluidDistances'][neighbors[0]!=neighbors[1]]

        dotProducts = torch.einsum('nd, nd -> n', distances, gradients)
        scattered = scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='max')
        simulationState['angleMin'] = torch.arccos(scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='min'))
        simulationState['angleMax'] = torch.arccos(scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='max'))

        if hasattr(self, 'boundaryModule') and simulation.boundaryModule.active:
            bb, bf = simulation.boundaryModule.boundaryToFluidNeighbors

            dotProducts = torch.einsum('nd, nd -> n', simulation.boundaryModule.boundaryToFluidNeighborDistances, simulationState['fluidNormal'][bf])
            scattered2 = scatter(dotProducts, bf, dim = 0, dim_size = simulationState['numParticles'], reduce='max')
            #             debugPrint(scattered.shape)
            #             debugPrint(scattered2.shape)
            scattered = torch.max(scattered, scattered2)

        scattered = torch.arccos(scattered)


        #         scattered = torch.arccos(scattered)
        mask = scattered.new_zeros(scattered.shape)
        mask[ torch.logical_and(scattered > np.pi/6, simulationState['fluidLambda'] < 0.6)] = 1
        # mask[ simulationState['fluidLambda'] < 0.6] = 1
        mask2 = scatter(mask[j],i, dim=0, dim_size = mask.shape[0], reduce='max')
        mask3 = scatter(mask2[j],i, dim=0, dim_size = mask.shape[0], reduce='max')
        finalMask = scattered.new_zeros(scattered.shape)
        finalMask[mask2 > 0] = 2/3
        finalMask[mask  > 0] = 1

        zeroRegion = finalMask > 0.7
        surfRegion = torch.logical_and(simulationState['fluidLambda'] >= 0.5, finalMask > 0.)
        bulkRegion = finalMask < 0.1
        simulationState['fluidSurfaceMask'] = torch.clone(finalMask)
        simulationState['fluidSurfaceMask'][surfRegion] = 1
        simulationState['fluidSurfaceMask'][zeroRegion] = 0
        simulationState['fluidSurfaceMask'][bulkRegion] = 2
        
    def adjustShiftingAmount(self, simulationState, simulation):
        normal = simulationState['fluidNormal']
        shiftAmount = simulationState['shiftAmount']
        shiftLength = torch.linalg.norm(shiftAmount, axis = 1)
        shiftAmount[shiftLength > self.support * 0.05] = \
            shiftAmount[shiftLength > self.support * 0.05] / shiftLength[shiftLength > self.support * 0.05,None] * self.support * 0.05
        
        surfaceMask = simulationState['fluidSurfaceMask']

        normalOuter = torch.einsum('nu, nv -> nuv', normal, normal)
        idMatrix = torch.tensor([[1,0],[0,1]], dtype = normal.dtype, device = normal.device)
        normalShift = torch.matmul(idMatrix - normalOuter, shiftAmount.unsqueeze(2))[:,:,0]
        # normalShift = torch.einsum('nuv, nu -> nu',idMatrix - normalOuter, shiftAmount)
        zeroRegion = surfaceMask < 0.5
        surfRegion = surfaceMask < 1.5
        bulkRegion = surfaceMask > 1.5

        shiftAmount[surfRegion] = normalShift[surfRegion]
        shiftAmount[bulkRegion] = shiftAmount[bulkRegion]
        shiftAmount[zeroRegion] = 0

        # shiftLength = torch.linalg.norm(shiftAmount, axis = 1)

        # shiftAmount[shiftLength > self.support * 0.1] = shiftAmount[shiftLength > self.support * 0.1] / shiftLength[shiftLength > self.support * 0.1,None] * self.support * 0.1

        simulationState['fluidUpdate'] = shiftAmount


        simulationState['fluidSurfaceMask'][surfRegion] = 1
        simulationState['fluidSurfaceMask'][zeroRegion] = 0
        simulationState['fluidSurfaceMask'][bulkRegion] = 2
    def detectSurface(self, simulationState, simulation):
        self.computeNormalizationmatrix(simulationState, simulation)
        self.computeFluidNormal(simulationState, simulation)
        self.detectFluidSurface(simulationState, simulation)
        
    def shift(self, simulationState, simulation):
        # if 'fluidSurfaceMask' not in simulationState:
            # self.detectSurface(simulationState, simulation)
        self.computeShiftStep(simulationState, simulation)
        # self.adjustShiftingAmount(simulationState, simulation)