import torch
from diffSPH.v2.math import scatter_sum
from typing import Dict, Optional

@torch.jit.script 
def sphInterpolation(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], kernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int):                                                        # Ancillary information
    j = neighborhood[1]
    k = masses[1][j] / densities[1][j] * kernels
    kq = torch.einsum('n..., n -> n...', quantities[1][j], k)
    
    return scatter_sum(kq, neighborhood[0], dim = 0, dim_size = numParticles)

@torch.jit.script 
def sphDensityInterpolation(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], kernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int):                                                        # Ancillary information
    j = neighborhood[1]
    kq = masses[1][j] * kernels
    
    return scatter_sum(kq, neighborhood[0], dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphGradient(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int, type : str = 'naive'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]
    if type == 'symmetric':
        k = masses[1][j].view(-1,1) * gradKernels
        Ai = torch.einsum('n..., n -> n...', quantities[0][i], 1.0 / densities[0][i]**2)
        Aj = torch.einsum('n..., n -> n...', quantities[1][j], 1.0 / densities[1][j]**2)
        kq = torch.einsum('n... , nd -> n...d', Ai + Aj, k)

        return torch.einsum('n, n... -> n...', densities[0], scatter_sum(kq, i, dim = 0, dim_size = numParticles))
    elif type == 'difference':
        k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
        kq = torch.einsum('n... , nd -> n...d', quantities[1][j] - quantities[0][i], k)
    elif type == 'summation':
        k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
        kq = torch.einsum('n... , nd -> n...d', quantities[1][j] + quantities[0][i], k)
    else:
        k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
        kq = torch.einsum('n... , nd -> n...d', quantities[1][j], k)
    
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphDivergence(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int, type : str = 'naive', mode : str = 'div'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]

    assert quantities[0].dim() > 1, 'Cannot compute divergence on non vector fields!'
    assert (mode in ['div','dot']), 'Only supports div F and nabla dot F'

    if type == 'symmetric':
        k = masses[1][j].view(-1,1) * gradKernels
        Ai = torch.einsum('n..., n -> n...', quantities[0][i], 1.0 / densities[0][i]**2)
        Aj = torch.einsum('n..., n -> n...', quantities[1][j], 1.0 / densities[1][j]**2)
        q = Ai + Aj
            
        if mode == 'div':
            kq = torch.einsum('n...d, nd -> n...', q, k)
        else:
            kq = torch.einsum('nd..., nd -> n...', q, k)

        return torch.einsum('n, n... -> n...', densities[0], scatter_sum(kq, i, dim = 0, dim_size = numParticles))
        
    q = quantities[1][j]
    k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
    
    if type == 'difference':
        q = quantities[1][j] - quantities[0][i]
    elif type == 'summation':
        q = quantities[1][j] + quantities[0][i]
        
    if mode == 'div':
        kq = torch.einsum('n...d, nd -> n...', q, k)
    else:
        kq = torch.einsum('nd..., nd -> n...', q, k)
            
    
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphCurl(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int, type : str = 'naive'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]

    assert quantities[0].dim() > 1, 'Cannot compute curl on non vector fields!'
    assert gradKernels.shape[1] > 1, 'Cannot compute curl on one-dimensional fields!'

    if type == 'symmetric':
        k = masses[1][j].view(-1,1) * gradKernels
        Ai = torch.einsum('n..., n -> n...', quantities[0][i], 1.0 / densities[0][i]**2)
        Aj = torch.einsum('n..., n -> n...', quantities[1][j], 1.0 / densities[1][j]**2)
        q = Ai + Aj
            
        if quantities[1].dim() == 2:
            kq = q[:,1] * k[:,0] - q[:,0] * k[:,1]
        else:
            kq = torch.cross(q, k, dim = -1)        
        
        return torch.einsum('n, n... -> n...', densities[0], scatter_sum(kq, i, dim = 0, dim_size = numParticles))
        
    q = quantities[1][j]
    k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
    
    if type == 'difference':
        q = quantities[1][j] - quantities[0][i]
    elif type == 'summation':
        q = quantities[1][j] + quantities[0][i]
        
    if quantities[1].dim() == 2:
        kq = q[:,1] * k[:,0] - q[:,0] * k[:,1]
    else:
        kq = torch.cross(q, k, dim = -1)            
    
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphLaplacian(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        laplaceKernels : Optional[torch.Tensor],    
        rij: torch.Tensor, xij:  torch.Tensor, hij : torch.Tensor,
        numParticles : int, type : str = 'naive'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]

    if quantities[0].dim() > 2:
        grad = sphGradient(masses, densities, quantities, neighborhood, gradKernels, numParticles, type = 'difference')
        div = sphDivergence(masses, densities, (grad, grad), neighborhood, gradKernels, numParticles, type = 'difference', mode = 'div')
    if type == 'naive':     
        assert laplaceKernels is not None, 'Laplace Kernel Values required for naive sph Laplacian operation'
        if laplaceKernels is not None:   
            print('naive')
            lk = -(masses[1][j] / densities[1][j]) * laplaceKernels
            kq = torch.einsum('n, n... -> n...', lk, quantities[0][i] - quantities[1][j])
            # kq = torch.einsum('n, n... -> n...', lk, quantities[1][j])
        
            return scatter_sum(kq, i, dim = 0, dim_size = numParticles)
            
    quotient = (rij * hij + 1e-7 * hij)
    kernelApproximation = torch.linalg.norm(gradKernels, dim = -1) /  quotient
    kernelApproximation = torch.einsum('nd, nd -> n', gradKernels, -xij)/  quotient# * rij * hij
    
    Aij = quantities[0][i] - quantities[1][j]
    if quantities[1].dim() == 1:
        kq = -Aij * (masses[1][j] / densities[1][j]) * 2 * kernelApproximation
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)
    
    if type == 'conserving':
        dot = torch.einsum('nd, nd -> n', Aij, xij) 
        q = (masses[1][j] / densities[1][j]) * kernelApproximation * dot# * rij
        kq = -q.view(-1, 1) * xij 
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)
        
    if type == 'divergenceFree':
        dot = torch.einsum('nd, nd -> n', Aij, xij) / (rij * hij + 1e-7 * hij)
        q = 2 * (xij.shape[1] + 2) *  (masses[1][j] / densities[1][j]) * dot
        kq = q.view(-1, 1) * gradKernels
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)

    if type == 'dot':
        term = -(xij.shape[1] + 2) * torch.einsum('nd, nd -> n', Aij, xij).view(-1,1) * xij - Aij
        kq = term * (masses[1][j] / densities[1][j] * kernelApproximation).view(-1,1)
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)

    q = -2 * (masses[1][j] / densities[1][j]) * kernelApproximation
    kq = Aij * q.view(-1,1)
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script
def sphOperation(
        masses : tuple[torch.Tensor, torch.Tensor],                                                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                                                              # Tuple of particle densities for (i,j)
        quantities : tuple[torch.Tensor, torch.Tensor],                                                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], kernels : torch.Tensor, kernelGradients : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels and kernelGradients ij
        radialDistances : torch.Tensor, directions : torch.Tensor, supports : torch.Tensor,                         # Graph information of |x_j - x_i| / hij, (x_j - x_i) / |x_j - x_i| and hij
        numParticles : int,                                                                                         # Ancillary information
        operation : str = 'interpolate', gradientMode : str = 'symmetric', divergenceMode : str = 'div',
        kernelLaplacians : Optional[torch.Tensor] = None):           # Operation to perform
    if operation == 'density':
        return sphDensityInterpolation(masses, densities, quantities, neighborhood, kernels, numParticles)
    if operation == 'interpolate':
        return sphInterpolation(masses, densities, quantities, neighborhood, kernels, numParticles)
    if operation == 'gradient':
        return sphGradient(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode)
    if operation == 'divergence':
        return sphDivergence(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode, mode = divergenceMode)
    if operation == 'curl':
        return sphCurl(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode)
    if operation == 'laplacian':
        return sphLaplacian(masses, densities, quantities, neighborhood, kernelGradients, kernelLaplacians, radialDistances, directions, supports, numParticles, type = gradientMode)
    if operation == 'directLaplacian':
        grad = sphGradient(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode)
        div = sphDivergence(masses, densities, (grad, grad), neighborhood, kernelGradients, numParticles, type = gradientMode, mode = divergenceMode)
        return div
    

def sphOperationFluidState(fluidState, quantities : tuple[torch.Tensor, torch.Tensor], operation : str = 'interpolate', gradientMode : str = 'symmetric', divergenceMode : str = 'div'):
    if operation == 'density':
        return sphDensityInterpolation(
            (fluidState['fluidMasses'], fluidState['fluidMasses']), 
            (fluidState['fluidMasses'], fluidState['fluidMasses']),
            (fluidState['fluidMasses'], fluidState['fluidMasses']), 
            fluidState['fluidNeighborhood']['indices'], 
            fluidState['fluidNeighborhood']['kernels'], 
            fluidState['numParticles'])
    return sphOperation(
        (fluidState['fluidMasses'], fluidState['fluidMasses']), 
        (fluidState['fluidDensities'], fluidState['fluidDensities']) if 'fluidDensities' in fluidState else (None, None),
        quantities, 
        fluidState['fluidNeighborhood']['indices'], 
        fluidState['fluidNeighborhood']['kernels'], fluidState['fluidNeighborhood']['gradients'], 
        fluidState['fluidNeighborhood']['distances'], fluidState['fluidNeighborhood']['vectors'], fluidState['fluidNeighborhood']['supports'], 
        fluidState['numParticles'], 
        operation = operation, gradientMode = gradientMode, divergenceMode = divergenceMode, 
        kernelLaplacians = fluidState['fluidNeighborhood']['laplacians'] if 'laplacians' in fluidState['fluidNeighborhood'] else None)