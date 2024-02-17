import torch
from diffSPH.v2.math import scatter_sum
from diffSPH.v2.neighborhood import neighborSearch
from diffSPH.v2.sphOps import sphOperation

def evalKernel(rij, xij, hij, k, dim):
    K = k.kernel(rij, hij, dim)
    J = k.Jacobi(rij, xij, hij, dim)
    H = k.Hessian2D(rij, xij, hij, dim)

    return K, J, H

@torch.jit.script
def LinearCG(H, B, x0, i, j, tol : float =1e-5):    
    xk = x0
    rk = torch.zeros_like(x0)
    numParticles = rk.shape[0] // 2

    rk[::2]  += scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
    rk[::2]  += scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

    rk[1::2] += scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
    rk[1::2] += scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)
    
    rk = rk - B
    
    pk = -rk
    rk_norm = torch.linalg.norm(rk)
    
    num_iter = 0
    while rk_norm > tol and num_iter < 32:
        apk = torch.zeros_like(x0)

        apk[::2]  += scatter_sum(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[::2]  += scatter_sum(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        apk[1::2] += scatter_sum(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[1::2] += scatter_sum(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        rkrk = torch.dot(rk, rk)
        
        alpha = rkrk / torch.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = torch.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1

        rk_norm = torch.linalg.norm(rk)
    return xk

@torch.jit.script
def BiCGStab(H, B, x0, i, j, tol : float =1e-5):
    xk = x0
    rk = torch.zeros_like(x0)
    numParticles = rk.shape[0] // 2

    rk[::2]  += scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
    rk[::2]  += scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

    rk[1::2] += scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
    rk[1::2] += scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)
    
    rk = B - rk
    r0 = rk.clone()
    pk = rk.clone()
    
    num_iter = 0
    while torch.linalg.norm(rk) > tol and num_iter < 32:
        apk = torch.zeros_like(x0)

        apk[::2]  += scatter_sum(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[::2]  += scatter_sum(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        apk[1::2] += scatter_sum(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[1::2] += scatter_sum(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        alpha = torch.dot(rk, r0) / torch.dot(apk, r0)
        sk = rk - alpha * apk
        ask = torch.zeros_like(x0)

        ask[::2]  += scatter_sum(H[:,0,0] * sk[j * 2], i, dim=0, dim_size=numParticles)
        ask[::2]  += scatter_sum(H[:,0,1] * sk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        ask[1::2] += scatter_sum(H[:,1,0] * sk[j * 2], i, dim=0, dim_size=numParticles)
        ask[1::2] += scatter_sum(H[:,1,1] * sk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        omega = torch.dot(ask, sk) / torch.dot(ask, ask)
        xk = xk + alpha * pk + omega * sk
        rk = sk - omega * ask

        beta = (torch.dot(rk, r0) / torch.dot(r0, r0)) * (alpha / omega)
        pk = rk + beta * (pk - omega * apk)
        
        num_iter += 1

    return xk

def computeShifting(p, areas, h, k, dim, periodic, domainMin, domainMax, computeRho = False, BiCG = True):
    numParticles = p.shape[0]
    i, j, rij, xij, hij, Wij, gradWij = neighborSearch(p, p, h, h, kernel = k, dim = dim, periodic = periodic, minDomain = domainMin, maxDomain = domainMax)

    K, J, H = evalKernel(rij, xij, hij, k, dim)
    if computeRho:
        ones = torch.ones_like(areas)
        rho = sphOperation((areas, areas), (ones, ones), (ones, ones), (i, j), Wij, gradWij, xij, rij, hij, p.shape[0], operation = 'interpolate')
        omega =  areas / rho
    else:
        omega = areas
    
    J = scatter_sum(J * omega[j,None], i, dim = 0, dim_size = numParticles)
    H = H * omega[j,None,None]

    B = torch.zeros(numParticles * 2, dtype = torch.float32)

    B[::2] = J[:,0]
    B[1::2] = J[:,1]
    x0 = torch.rand(numParticles * 2).to(p.device).type(p.dtype) * h / 4 - h / 8

    if BiCG:
        xk = BiCGStab(H, B, x0, i, j)
    else:
        xk = LinearCG(H, B, x0, i, j)
    # print(f'xk: {xk} ({xk.shape})')
    return torch.vstack((-xk[::2],-xk[1::2])).T , J, H, B