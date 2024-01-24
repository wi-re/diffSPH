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

from ..kernels import kernel, spikyGrad, kernelGradient
from ..module import Module
from ..parameter import Parameter
from ..util import *

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

@torch.jit.script
def computeNormalizationMatrix(i, j, ri, rj, Vi, Vj, distances, radialDistances, support : float, numParticles : int, eps : float):
    gradW = kernelGradient(radialDistances, distances, support)

    # r_ba = rj[j] - ri[i]
    r_ba = -distances * radialDistances[:,None] * support
    fac = Vj[j] * 2

    term = torch.einsum('nu,nv -> nuv', r_ba, gradW)
    term[:,0,0] = term[:,0,0] * fac
    term[:,0,1] = term[:,0,1] * fac
    term[:,1,0] = term[:,1,0] * fac
    term[:,1,1] = term[:,1,1] * fac

    fluidNormalizationMatrix = scatter_sum(term, i, dim=0, dim_size=numParticles)
    return fluidNormalizationMatrix

@torch.jit.script 
def computeRenormalizedDensityGradient(i, j, ri, rj, Vi, Vj, distances, radialDistances, support : float, numParticles : int, eps : float, Li, Lj, rhoi, rhoj, grad):
    gradW = kernelGradient(radialDistances, distances, support)
    
    rho_ba = rhoj[j] - rhoi[i] 
    # grad = torch.matmul(Li[i], gradW.unsqueeze(2))[:,:,0]

    gradMagnitude = torch.linalg.norm(grad, dim=1)
    kernelMagnitude = torch.linalg.norm(gradW, dim=1)        
    change =  torch.abs(gradMagnitude - kernelMagnitude) / (kernelMagnitude + eps)
    # grad[change > 0.1,:] = gradW[change > 0.1, :]
    # grad = gradW

    renormalizedGrad = grad
    renormalizedDensityGradient = scatter_sum((rho_ba * Vj[j] * 2)[:,None] * grad, i, dim = 0, dim_size=numParticles)
    
    return renormalizedGrad, renormalizedDensityGradient

@torch.jit.script
def computeDensityDiffusionDeltaSPH(i, j, ri, rj, Vi, Vj, distances, radialDistances, support : float, numParticles : int, eps : float, gradRhoi, gradRhoj, rhoi, rhoj, delta : float, c0 : float):
    gradW = kernelGradient(radialDistances, distances, support)
    rji = -distances * radialDistances[:,None] * support
    # rji = ri[j] - rj[i]
    rji2 = torch.linalg.norm(rji, dim=1)**2 + eps

    # delta SPH Term
    # densityTerm = -(gradRhoi[i] + gradRhoj[j])
    # diffusionTerm = (2 * (rhoj[j] - rhoi[i]) / rji2)[:,None] * rji

    # psi_ij = diffusionTerm + densityTerm
    # prod = torch.einsum('nu,nu -> n', psi_ij, gradW) 
    # return support * delta * c0 * scatter_sum(prod * Vj[j], i, dim=0, dim_size = numParticles)

    # Antuono correction
    densityTerm = 0.5 * torch.einsum('nu,nu -> n', gradRhoi[i] + gradRhoj[j], rji)
    diffusionTerm = (rhoj[j] - rhoi[i])
    gradTerm = torch.einsum('nu,nu -> n', gradW, rji / rji2[:,None])
    prod = (diffusionTerm + densityTerm) * gradTerm

    return 2 * support * delta * c0 * scatter_sum(prod * Vj[j], i, dim=0, dim_size = numParticles)

# Molteni and Colagrossi - 2009 - A simple procedure to improve the pressure evaluation in hydrodynamic context using the SPH
@torch.jit.script
def computeDensityDiffusionMOG(i, j, ri, rj, Vi, Vj, distances, radialDistances, support : float, numParticles : int, eps : float, rhoi, rhoj, delta : float, c0 : float):
    gradW = kernelGradient(radialDistances, distances, support)
    rji = -distances * radialDistances[:,None] * support # ri[j] - rj[i]
    rji2 = torch.linalg.norm(rji, dim=1)**2 + eps

    psi_ij = (2 * (rhoj[j] - rhoi[i]) / rji2)[:,None] * rji
    prod = torch.einsum('nu,nu -> n', psi_ij, gradW) 
    return support * delta * c0 * scatter_sum(prod * Vj[j], i, dim=0, dim_size = numParticles)


from ..util import pinv2x2
def invertNormalizationMatrix_PINV(normalizationMatrix, gradW, i):
    Li, _ = pinv2x2(normalizationMatrix)
    grad = torch.matmul(Li[i], gradW.unsqueeze(2))[:,:,0]
    return grad

def batch_gj_solve(m, n, nb, result):
    i, j, eqns, colrange, augCol, col, row, bigrow, nt = 0,0,0,0,0,0,0,0,0
    eqns = n
    colrange = n
    augCol = n + nb
    nt = n + nb
    nElems = m.shape[0]

    for col in range(colrange):
        bigrow = torch.ones(nElems).type(torch.long).to(m.device) * col
        for row in range(col + 1, colrange):
            selected = torch.gather(m,1,(nt * bigrow + col)[:,None])[:,0]
            condition = abs(m[:,nt*row + col]) > selected
            bigrow = torch.where(condition, row, bigrow)
            temp = m[:,nt * row + col]
            gathered = torch.gather(m,1,(nt * bigrow + col)[:,None])[:,0]
            m[:, nt * row + col] = torch.where(condition, gathered, m[:, nt * row + col])
            torch.scatter(m, 1, (nt * bigrow + col)[:,None], torch.where(condition,temp, gathered)[:,None])

    rr, rrcol, rb, rbr, kup, kupr, kleft, kleftr = 0,0,0,0,0,0,0,0

    retVal = torch.zeros(nElems, device = m.device)

    for rrcol in range(0, colrange):
        for rr in range(rrcol + 1, eqns):
            dnr = m[:,nt*rrcol + rrcol]
            condition = abs(dnr) >= 1e-12
            retVal[torch.logical_not(condition)] = 1
            cc = -m[condition,nt*rr + rrcol] / dnr[condition]
            for j in range(augCol):
                m[condition,nt*rr + j] = m[condition,nt*rr + j] + cc * m[condition,nt*rrcol + j]

                
    backCol, backColr = 0,0
    tol = 1.0e-12
    for rbr in range(eqns):
        rb = eqns - rbr - 1
        condA = m[:,nt*rb + rb] == 0
        condB = abs(m[:,nt*rb + augCol - 1]) > tol
        retVal[torch.logical_and(condA, condB)] = 1
        condition = torch.logical_not(condA)

        for backColr in range(rb, augCol):
            backCol = rb + augCol - backColr - 1
            m[condition, nt*rb + backCol] = m[condition, nt*rb + backCol] / m[condition, nt*rb + rb]
        if not (rb == 0):
            for kupr in range(rb):
                kup = rb - kupr - 1
                for kleftr in range(rb, augCol):
                    kleft = rb + augCol - kleftr - 1
                    kk = -m[condition, nt*kup + rb] / m[condition, nt*rb + rb]
                    m[condition, nt*kup + kleft] = (m[condition, nt*kup + kleft] +
                                         kk * m[condition, nt*rb + kleft])

    for i in range(n):
        for j in range(nb):
            result[retVal == 0, nb*i + j] = m[retVal == 0,nt*i + n + j]
    return retVal


def invertNormalizationMatrix_GJ(normalizationMatrix, gradW, ni):
    expanded = torch.zeros((normalizationMatrix.shape[0],3,3))
    expanded[:,:2,:2] = normalizationMatrix
    n = 2
    nt = n + 1
    expanded = expanded.reshape(normalizationMatrix.shape[0],9)[ni,:]

    temp = torch.zeros((ni.shape[0], 12), dtype = gradW.dtype, device = gradW.device)
    res  = torch.zeros((ni.shape[0], 3), dtype = gradW.dtype, device = gradW.device)
    for i in range(n):
        for j in range(n):
            temp[:,nt * i + j] = expanded[:,3 * i + j]
        temp[:, nt*i + n] = gradW[:,i]
    m = torch.clone(temp)
    n = n
    nb = 1
    result = torch.zeros((ni.shape[0], 3), dtype = gradW.dtype, device = gradW.device)
    m = torch.clone(temp)
    res = torch.zeros((ni.shape[0], 3), dtype = gradW.dtype, device = gradW.device)
    batch_gj_solve(m, n, 1, res)

    return res[:,:n]

class densityDiffusionModule(Module):
    def getParameters(self):
        return [
            # Parameter('deltaSPH', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'delta', 'float', 0.1, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'c0', 'float', -1.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'beta', 'float', -1.0, required = False, export = True, hint = ''),
            Parameter('deltaSPH', 'HughesGrahamCorrection', 'bool', True, required = False, export = True, hint = '')
        ]
        
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
        dx = simulationConfig['particle']['support'] * simulationConfig['particle']['packing']
        c0 = 10.0 * np.sqrt(2.0*9.81*0.3)
        h0 = simulationConfig['particle']['support']
        dt = 0.25 * h0 / (1.1 * c0)
        if simulationConfig['fluid']['c0'] < 0:
            simulationConfig['fluid']['c0'] = c0
        
        self.c0 = simulationConfig['fluid']['c0']
        self.eps = self.support **2 * 0.1
        self.scheme = simulationConfig['diffusion']['densityScheme']

    def resetState(self, simulationState):
        self.normalizationMatrix = None
        self.fluidL = None
        self.eigVals = None
        self.densityDiffusion = None

    def computeNormalizationMatrices(self, simulationState, simulation):
        with record_function('density[diffusion] - compute normalization matrices'):
            self.normalizationMatrix = computeNormalizationMatrix(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps)     
            # self.normalizationMatrix += simulation.boundaryModule.computeNormalizationMatrices(simulationState, simulation)
            # self.fluidL, self.eigVals = pinv2x2(self.normalizationMatrix)
    def computeRenormalizedDensityGradient(self, simulationState, simulation):
        with record_function('density[diffusion] - compute renormalized density gradient'):
            gradW = kernelGradient(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)    
            # normalizedGradients = invertNormalizationMatrix_GJ(self.normalizationMatrix, gradW, simulationState['fluidNeighbors'][0])
            normalizedGradients = invertNormalizationMatrix_PINV(self.normalizationMatrix, gradW, simulationState['fluidNeighbors'][0])
            dwij_mag = torch.linalg.norm(gradW, axis = 1, ord = 1)
            norm_mag = torch.linalg.norm(normalizedGradients, axis = 1, ord = 1)

            eps = 1e-4 * self.support
            tol = 0.1
            change = abs(norm_mag - dwij_mag) / (dwij_mag + eps)
            normalizedGradients = torch.where((change < tol)[:,None], normalizedGradients, gradW)
            # normalizedGradients = normalizedGradients

            self.renormalizedGrad, self.renormalizedDensityGradient = computeRenormalizedDensityGradient(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  self.fluidL, self.fluidL, simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity, normalizedGradients)     
            # self.renormalizedDensityGradient  += simulation.boundaryModule.computeRenormalizedDensityGradient(simulationState, simulation)
  
    def computeDensityDiffusion(self, simulationState, simulation):
        with record_function('density[diffusion] - compute density diffusion'):
            if self.scheme == 'deltaSPH':
                if 'fluidL' in simulationState:
                    self.normalizationMatrix = simulationState['normalizationMatrix']
                    self.fluidL = simulationState['fluidL']
                    self.eigVals = simulationState['eigVals']
                else:
                    self.computeNormalizationMatrices(simulationState, simulation)
                    simulationState['normalizationMatrix'] = self.normalizationMatrix
                    simulationState['fluidL'] = self.fluidL
                    simulationState['eigVals'] = self.eigVals

                self.computeRenormalizedDensityGradient(simulationState, simulation)
                self.densityDiffusion = computeDensityDiffusionDeltaSPH(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  self.renormalizedDensityGradient, self.renormalizedDensityGradient, \
                                                                                                  simulationState['fluidDensity'] * self.restDensity,simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  self.delta, self.c0)
                simulationState['dpdt'] += self.densityDiffusion
            elif self.scheme == 'MOG':
                self.densityDiffusion = computeDensityDiffusionMOG(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity,simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  self.delta, self.c0)
                simulationState['dpdt'] += self.densityDiffusion
            # simulation.sync(simulationState['dpdt'])

            # self.densityDiffusion += simulation.boundaryModule.computeDensityDiffusion(simulationState, simulation)