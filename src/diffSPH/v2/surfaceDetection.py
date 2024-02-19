import torch
from diffSPH.v2.math import scatter_sum, mod
import numpy as np
from diffSPH.v2.sphOps import sphOperation

from torch.profiler import profile, record_function, ProfilerActivity
from diffSPH.v2.math import pinv2x2

# # Barecasco et al 2013: Simple free-surface detection in two and three-dimensional SPH solver
# def detectFreeSurfaceBarecasco(threshold, i, j, Wij, gradWij, xij, rij, hij, particles, numParticles):
#     coverVector = scatter_sum(-xij, i, dim = 0, dim_size = numParticles)
#     normalized = torch.nn.functional.normalize(coverVector)
#     angle = torch.arccos(torch.einsum('ij,ij->i', xij, normalized[i]))
#     condition = (angle <= threshold / 2) & (i != j) | (torch.linalg.norm(normalized, dim = -1)[i] <= 1e-5)
#     fs = ~scatter_sum(condition, i, dim = 0, dim_size = numParticles) 
#     return fs, coverVector, normalized

# def detectFreeSurfaceUsingColorField(areas, rho, ones, i, j, Wij, gradWij, xij, rij, hij, particles, numParticles, targetNumNeighbors):
#     colorField = sphOperation((areas, areas), (rho, rho), (ones, ones), (i, j), Wij, gradWij, rij, xij, hij, particles.shape[0], operation = 'interpolate')
#     nj = scatter_sum(torch.ones_like(colorField[j]), i, dim = 0, dim_size = numParticles)
#     colorFieldMean = scatter_sum(colorField[j], i, dim = 0, dim_size = numParticles) / nj
#     fs = torch.where((colorField < colorFieldMean) & (nj < targetNumNeighbors * 0.8), 1., 0.)
#     return fs

# def detectFreeSurfaceUsingColorFieldGradient(threshold, areas, rho, ones, i, j, Wij, gradWij, xij, rij, hij, particles, numParticles):
#     colorField = sphOperation((areas, areas), (rho, rho), (ones, ones), (i, j), Wij, gradWij, rij, xij, hij, particles.shape[0], operation = 'interpolate')
#     gradColorField = sphOperation((areas, areas), (rho, rho), (colorField, colorField), (i, j), Wij, gradWij, rij, xij, hij, particles.shape[0], operation = 'gradient', gradientMode = 'difference')
#     fs = torch.linalg.norm(gradColorField, dim = -1) > threshold
#     return fs, colorField, gradColorField

# def computeNormalizationMatrices(particles, areas, rho, i, j, Wij, gradWij, rij, xij, hij, numParticles):
#     normalizationMatrices = -sphOperation((areas, areas), (rho, rho), (particles, particles), (i, j), Wij, gradWij, rij, xij, hij, numParticles, operation = 'gradient', gradientMode = 'difference')

#     L, lambdas = pinv2x2(normalizationMatrices)

#     return L, normalizationMatrices, lambdas

# def computeNormals(particles, lambdas, areas, ones, rho, i, j, Wij, gradWij, xij, rij, hij, numParticles, L):
#     term = sphOperation((areas, areas), (rho, rho), (ones, ones), (i, j), Wij, gradWij, xij, rij, hij, particles.shape[0], operation = 'gradient', gradientMode='naive')

#     nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
#     n = torch.nn.functional.normalize(nu, dim = -1)
#     lMin = torch.min(torch.abs(lambdas), dim = -1).values

#     return n, lMin

# # See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
# def detectFreeSurfaceMaronne(n, i, j, rij, xij, hij, particles, supports, numParticles, periodic = False, domainMin = None, domainMax = None):
#     T = particles + n * supports.view(-1,1) / 3

#     tau = torch.vstack((-n[:,1], n[:,0])).mT
#     xjt = particles[j] - T[i]
#     xjt = torch.stack([xjt[:,i] if not periodic else mod(xjt[:,i], domainMin[i], domainMax[i]) for i in range(xjt.shape[1])], dim = -1)

#     condA1 = rij >= np.sqrt(2) * hij / 3
#     condA2 = torch.linalg.norm(xjt, dim = -1) <= hij / 3
#     condA = (condA1 & condA2) & (i != j)
#     cA = scatter_sum(condA, i, dim = 0, dim_size = numParticles)

#     condB1 = rij < np.sqrt(2) * hij / 3
#     condB2 = torch.abs(torch.einsum('ij,ij->i', -n[i], xjt)) + torch.abs(torch.einsum('ij,ij->i', tau[i], xjt)) < hij / 3
#     condB = (condB1 & condB2) & (i != j)
#     cB = scatter_sum(condB, i, dim = 0, dim_size = numParticles)
    
#     fs = torch.where(~cA & ~cB, 1.,0.)
#     return fs, cA, cB
# def expandFreeSurfaceMask(fs, i, j, numParticles):
#     fsm = scatter_sum(fs[j], i, dim = 0, dim_size = numParticles)
#     return fsm > 0