import torch
from diffSPH.v2.math import scatter_sum, mod
import numpy as np
from diffSPH.v2.sphOps import sphOperation

from torch.profiler import profile, record_function, ProfilerActivity

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


# Barecasco et al 2013: Simple free-surface detection in two and three-dimensional SPH solver
def detectFreeSurfaceBarecasco(threshold, i, j, Wij, gradWij, xij, rij, hij, particles, numParticles):
    coverVector = scatter_sum(-xij, i, dim = 0, dim_size = numParticles)
    normalized = torch.nn.functional.normalize(coverVector)
    angle = torch.arccos(torch.einsum('ij,ij->i', xij, normalized[i]))
    condition = (angle <= threshold / 2) & (i != j) | (torch.linalg.norm(normalized, dim = -1)[i] <= 1e-5)
    fs = ~scatter_sum(condition, i, dim = 0, dim_size = numParticles) 
    return fs, coverVector, normalized

def detectFreeSurfaceUsingColorField(areas, rho, ones, i, j, Wij, gradWij, xij, rij, hij, particles, numParticles, targetNumNeighbors):
    colorField = sphOperation((areas, areas), (rho, rho), (ones, ones), (i, j), Wij, gradWij, rij, xij, hij, particles.shape[0], operation = 'interpolate')
    nj = scatter_sum(torch.ones_like(colorField[j]), i, dim = 0, dim_size = numParticles)
    colorFieldMean = scatter_sum(colorField[j], i, dim = 0, dim_size = numParticles) / nj
    fs = torch.where((colorField < colorFieldMean) & (nj < targetNumNeighbors * 0.8), 1., 0.)
    return fs

def detectFreeSurfaceUsingColorFieldGradient(threshold, areas, rho, ones, i, j, Wij, gradWij, xij, rij, hij, particles, numParticles):
    colorField = sphOperation((areas, areas), (rho, rho), (ones, ones), (i, j), Wij, gradWij, rij, xij, hij, particles.shape[0], operation = 'interpolate')
    gradColorField = sphOperation((areas, areas), (rho, rho), (colorField, colorField), (i, j), Wij, gradWij, rij, xij, hij, particles.shape[0], operation = 'gradient', gradientMode = 'difference')
    fs = torch.linalg.norm(gradColorField, dim = -1) > threshold
    return fs, colorField, gradColorField

def computeNormalizationMatrices(particles, areas, rho, i, j, Wij, gradWij, rij, xij, hij, numParticles):
    normalizationMatrices = -sphOperation((areas, areas), (rho, rho), (particles, particles), (i, j), Wij, gradWij, rij, xij, hij, numParticles, operation = 'gradient', gradientMode = 'difference')

    L, lambdas = pinv2x2(normalizationMatrices)

    return L, normalizationMatrices, lambdas

def computeNormals(particles, lambdas, areas, ones, rho, i, j, Wij, gradWij, xij, rij, hij, numParticles, L):
    term = sphOperation((areas, areas), (rho, rho), (ones, ones), (i, j), Wij, gradWij, xij, rij, hij, particles.shape[0], operation = 'gradient', gradientMode='naive')

    nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
    n = torch.nn.functional.normalize(nu, dim = -1)
    lMin = torch.min(torch.abs(lambdas), dim = -1).values

    return n, lMin

# See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
def detectFreeSurfaceMaronne(n, i, j, rij, xij, hij, particles, supports, numParticles, periodic = False, domainMin = None, domainMax = None):
    T = particles + n * supports.view(-1,1) / 3

    tau = torch.vstack((-n[:,1], n[:,0])).mT
    xjt = particles[j] - T[i]
    xjt = torch.stack([xjt[:,i] if not periodic else mod(xjt[:,i], domainMin[i], domainMax[i]) for i in range(xjt.shape[1])], dim = -1)

    condA1 = rij >= np.sqrt(2) * hij / 3
    condA2 = torch.linalg.norm(xjt, dim = -1) <= hij / 3
    condA = (condA1 & condA2) & (i != j)
    cA = scatter_sum(condA, i, dim = 0, dim_size = numParticles)

    condB1 = rij < np.sqrt(2) * hij / 3
    condB2 = torch.abs(torch.einsum('ij,ij->i', -n[i], xjt)) + torch.abs(torch.einsum('ij,ij->i', tau[i], xjt)) < hij / 3
    condB = (condB1 & condB2) & (i != j)
    cB = scatter_sum(condB, i, dim = 0, dim_size = numParticles)
    
    fs = torch.where(~cA & ~cB, 1.,0.)
    return fs, cA, cB
def expandFreeSurfaceMask(fs, i, j, numParticles):
    fsm = scatter_sum(fs[j], i, dim = 0, dim_size = numParticles)
    return fsm > 0