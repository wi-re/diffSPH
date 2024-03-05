import torch
from diffSPH.v2.math import scatter_sum
# from diffSPH.v2.neighborhood import neighborSearch
from diffSPH.v2.sphOps import sphOperation

def evalKernel(rij, xij, hij, k, dim):
    K = k.kernel(rij, hij, dim)
    J = k.Jacobi(rij, xij, hij, dim)
    H = k.Hessian2D(rij, xij, hij, dim)

    return K, J, H

@torch.jit.script
def LinearCG(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):    
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
    
    convergence = []
    num_iter = 0
    rk_norm = torch.linalg.norm(rk)
    while (torch.abs(torch.linalg.norm(rk) / rk_norm - 1) > 1e-4 or num_iter == 0) and num_iter < maxIter and torch.linalg.norm(rk) > tol:
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

        # rk_norm = torch.linalg.norm(rk)
        # print(f'Iter: {num_iter}, Residual: {torch.linalg.norm(rk)}, Threshold {tol}, Ratio {torch.linalg.norm(rk) / rk_norm - 1}')

        convergence.append(torch.linalg.norm(rk))

    return xk, convergence, num_iter, torch.linalg.norm(rk)

@torch.jit.script
def BiCGStab(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):
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

    convergence = []
    rk_norm = torch.linalg.norm(rk)
    while (torch.abs(torch.linalg.norm(rk) / rk_norm - 1) > 1e-4 or num_iter == 0) and num_iter < maxIter and torch.linalg.norm(rk) > tol:
    # while num_iter < maxIter and torch.linalg.norm(rk) > tol:
        rk_norm = torch.linalg.norm(rk)
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
        # rk_norm = torch.linalg.norm(rk)
        # print(f'Iter: {num_iter}, Residual: {torch.linalg.norm(rk)}, Threshold {tol}, Ratio {torch.linalg.norm(rk) / rk_norm - 1}')

        convergence.append(torch.linalg.norm(rk))

    return xk, convergence, num_iter, torch.linalg.norm(rk)

def deltaPlusShifting(particleState, config):
    W_0 = config['kernel']['function'].kernel(torch.tensor(config['particle']['dx'] / config['particle']['support'] * config['kernel']['kernelScale']), torch.tensor(config['particle']['support']), dim = config['domain']['dim'])
    W_0 = config['kernel']['function'].kernel(torch.tensor(config['particle']['dx'] / config['particle']['support']), torch.tensor(config['particle']['support']), dim = config['domain']['dim'])

    (i,j) = particleState['fluidNeighborhood']['indices']
    k = particleState['fluidNeighborhood']['kernels'] / W_0
    gradK = particleState['fluidNeighborhood']['gradients']

    R = config['shifting']['R']
    n = config['shifting']['n']
    term = (1 + R * torch.pow(k, n))
    densityTerm = particleState['fluidMasses'][j] / (particleState['fluidDensities'][i] + particleState['fluidDensities'][j])
    phi_ij = 1

    scalarTerm = term * densityTerm * phi_ij
    shiftAmount = scatter_sum(scalarTerm.view(-1,1) * gradK, i, dim = 0, dim_size = particleState['fluidPositions'].shape[0])

    CFL = config['shifting']['CFL']
    if config['shifting']['computeMach'] == False:
        Ma = 0.1
    else:
        Ma = torch.amax(torch.linalg.norm(particleState['fluidVelocities'], dim = -1)) / config['fluid']['cs']
    shiftScaling = -CFL * Ma * (particleState['fluidSupports'] / config['kernel']['kernelScale'] * 2)**2
    # print(particleState['fluidSupports'])
    return shiftScaling.view(-1,1) * shiftAmount



def computeLambdaGrad(simulationState, config):    
    (i, j) = simulationState['fluidNeighborhood']['indices']

    gradKernel = simulationState['fluidNeighborhood']['gradients']
    Ls = simulationState['fluidL'][i]

    normalizedGradients = torch.einsum('ijk,ik->ij', Ls, gradKernel)

    return torch.nn.functional.normalize(sphOperation(
        (simulationState['fluidMasses'], simulationState['fluidMasses']), 
        (simulationState['fluidDensities'], simulationState['fluidDensities']),
        (simulationState['fluidLambdas'], simulationState['fluidLambdas']),
        simulationState['fluidNeighborhood']['indices'], 
        simulationState['fluidNeighborhood']['kernels'], normalizedGradients,
        simulationState['fluidNeighborhood']['distances'], simulationState['fluidNeighborhood']['vectors'], simulationState['fluidNeighborhood']['supports'], 
        simulationState['numParticles'], 
        operation = 'gradient', gradientMode = 'difference', divergenceMode = 'div', 
        kernelLaplacians = simulationState['fluidNeighborhood']['laplacians'] if 'laplacians' in simulationState['fluidNeighborhood'] else None), dim = -1)

from diffSPH.v2.sphOps import sphOperationFluidState
from diffSPH.v2.modules.neighborhood import fluidNeighborSearch

@torch.jit.script
def BiCG(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):
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
    convergence = []
    rk_norm = torch.linalg.norm(rk)
    while (torch.abs(torch.linalg.norm(rk) / rk_norm - 1) > 1e-4 or num_iter == 0) and num_iter < maxIter and torch.linalg.norm(rk) > tol:
        rk_norm = torch.linalg.norm(rk)
        apk = torch.zeros_like(x0)

        apk[::2]  += scatter_sum(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[::2]  += scatter_sum(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        apk[1::2] += scatter_sum(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[1::2] += scatter_sum(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        alpha = torch.dot(rk, r0) / torch.dot(apk, r0)
        xk = xk + alpha * pk
        rk = rk - alpha * apk

        beta = torch.dot(rk, r0) / torch.dot(r0, r0)
        pk = rk + beta * pk
        
        num_iter += 1
        convergence.append(torch.linalg.norm(rk))

    return xk, convergence, num_iter, torch.linalg.norm(rk)

# @torch.jit.script
def BiCGStab_wJacobi(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):
    xk = x0
    rk = torch.zeros_like(x0)
    numParticles = rk.shape[0] // 2

    # Calculate the Jacobi preconditioner
    diag = torch.vstack((H[i == j, 0, 0], H[i == j, 1, 1])).flatten()
    # diag[diag < 1e-8] = 1
    M_inv = 1 / diag
    M_inv[diag < 1e-8] = 0
    # M_inv[torch.isnan(M_inv)] = 1

    rk[::2]  += scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
    rk[::2]  += scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

    rk[1::2] += scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
    rk[1::2] += scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)
    
    rk = B - rk
    r0 = rk.clone()

    # Apply the preconditioner
    # zk = torch.bmm(M_inv, rk.unsqueeze(-1)).squeeze(-1)
    # print(f'i: {i.shape} x j: {j.shape} -> {i[i==j]}, {i}, {j}')
    # print(f'rk: {rk}, M_inv: {M_inv}, @ {M_inv.shape} * {rk.shape}')
    zk = M_inv * rk
    pk = zk.clone()
    
    num_iter = 0
    convergence = []
    rk_norm = torch.linalg.norm(rk)
    while (torch.abs(torch.linalg.norm(rk) / rk_norm - 1) > 1e-3 or num_iter == 0) and num_iter < maxIter and torch.linalg.norm(rk) > tol:
        rk_norm = torch.linalg.norm(rk)
        apk = torch.zeros_like(x0)

        apk[::2]  += scatter_sum(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[::2]  += scatter_sum(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        apk[1::2] += scatter_sum(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
        apk[1::2] += scatter_sum(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        alpha = torch.dot(rk, r0) / (torch.dot(apk, r0) + 1e-8)
        sk = rk - alpha * apk
        ask = torch.zeros_like(x0)

        ask[::2]  += scatter_sum(H[:,0,0] * sk[j * 2], i, dim=0, dim_size=numParticles)
        ask[::2]  += scatter_sum(H[:,0,1] * sk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        ask[1::2] += scatter_sum(H[:,1,0] * sk[j * 2], i, dim=0, dim_size=numParticles)
        ask[1::2] += scatter_sum(H[:,1,1] * sk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        omega = torch.dot(ask, sk) / (torch.dot(ask, ask) + 1e-8)
        xk = xk + alpha * pk + omega * sk
        rk = sk - omega * ask

        # Apply the preconditioner
        zk = M_inv * rk
        beta = (torch.dot(rk, r0) / (torch.dot(r0, r0)) + 1e-8) * (alpha / (omega + 1e-8))
        pk = zk + beta * (pk - omega * apk)
        if torch.abs(alpha) < 1e-8 or torch.abs(omega) < 1e-8 or torch.abs(beta) < 1e-8:
            break

        # print('###############################################################################')
        # print(f'Iter: {num_iter}, Residual: {torch.linalg.norm(rk)}, Threshold {tol}')
        # print(f'alpha: {alpha}, omega: {omega}, beta: {beta}')
        # print(f'rk: {rk}, pk: {pk}, xk: {xk}')
        # print(f'apk: {apk}, ask: {ask}')
        # print(torch.dot(rk, r0))
        # print(torch.dot(r0, r0))
        # print((alpha / omega))

        num_iter += 1
        convergence.append(torch.linalg.norm(rk))

    return xk, convergence, num_iter, torch.linalg.norm(rk)

def computeShifting(particleState, config, computeRho = False, scheme = 'BiCG'):
    numParticles = particleState['fluidPositions'].shape[0]
    if config['shifting']['freeSurface']:  
        if config['shifting']['useExtendedMask']:
            fs = particleState['fluidFreeSurfaceMask']
        else:
            fs = particleState['fluidFreeSurface']
        fsm = particleState['fluidFreeSurfaceMask']

    # particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
    (i,j) = particleState['fluidNeighborhood']['indices']
    rij = particleState['fluidNeighborhood']['distances']
    xij = particleState['fluidNeighborhood']['vectors']
    hij = particleState['fluidNeighborhood']['supports']
    k = config['kernel']['function']
    dim = config['domain']['dim']

    K, J, H = evalKernel(rij, xij, hij, k, dim)
    if computeRho:
        particleState['fluidDensities'] = sphOperationFluidState(particleState, None, 'density')
        omega =  particleState['fluidMasses'] / particleState['fluidDensities']
    else:
        omega = particleState['fluidAreas']
    
    
    J = scatter_sum(J * omega[j,None], i, dim = 0, dim_size = numParticles)
    H = H * omega[j,None,None]

    h2 = particleState['fluidSupports'].repeat(2,1).T.flatten()
    x0 = torch.rand(numParticles * 2).to(rij.device).type(rij.dtype) * config['particle']['dx'] / 4 - config['particle']['dx'] / 8
    if config['shifting']['initialization'] == 'deltaPlus':
        x0 = -deltaPlusShifting(particleState, config).flatten() * 0.5
    if config['shifting']['initialization'] == 'deltaMinus':
        x0 = deltaPlusShifting(particleState, config).flatten() * 0.5
    if config['shifting']['initialization'] == 'zero':
        x0 = torch.zeros_like(x0)
    

    B = torch.zeros(numParticles * 2, dtype = torch.float32, device=rij.device)
    if config['shifting']['freeSurface']:

        J2 = torch.zeros(J.shape[0], 2, dtype = torch.float32, device=rij.device)
        J2[fs < 0.5, :] = J[fs < 0.5, :]

        B[::2] = J2[:,0]
        B[1::2] = J2[:,1]

        x0 = x0.view(-1,2)
        x0[fs > 0.5,0] = 0
        x0[fs > 0.5,1] = 0
        x0 = x0.flatten()
        H[fs[i] > 0.5,:,:] = 0
        H[fs[j] > 0.5,:,:] = 0
        iMasked = i[fs[i] < 0.5]
        jMasked = j[fs[i] < 0.5]
        if scheme == 'BiCG':
            xk, convergence, iters, residual = BiCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab':
            xk, convergence, iters, residual = BiCGStab(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab_wJacobi':
            xk, convergence, iters, residual = BiCGStab_wJacobi(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        else:
            xk, convergence, iters, residual = LinearCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
        # print(f'Iter: {iters}, Residual: {residual}, fs: xk fs: {update[fs > 0.5]}')
        # if scheme == 'BiCG':
        #     xk = BiCG(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        # elif scheme == 'BiCGStab':
        #     xk = BiCGStab(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        # elif scheme == 'BiCGStab_wJacobi':
        #     xk = BiCGStab_wJacobi(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        # else:
        #     xk = LinearCG(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
    else:
        B[::2] = J[:,0]
        B[1::2] = J[:,1]
        if scheme == 'BiCG':
            xk, convergence, iters, residual = BiCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab':
            xk, convergence, iters, residual = BiCGStab(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab_wJacobi':
            xk, convergence, iters, residual = BiCGStab_wJacobi(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        else:
            xk, convergence, iters, residual = LinearCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])

    update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
    return update, K, J, H, B, convergence, iters, residual

from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne, expandFreeSurfaceMask, computeColorField, computeColorFieldGradient, detectFreeSurfaceColorFieldGradient

def solveShifting(particleState, config):
    initialPositions = torch.clone(particleState['fluidPositions'])
    initialDensities = torch.clone(particleState['fluidDensities'])
    overallStates = []
    for i in range(config['shifting']['maxIterations']):
        particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
        if config['shifting']['summationDensity']:
            particleState['fluidDensities'] = sphOperationFluidState(particleState, None, 'density')

        if config['shifting']['freeSurface']:
            if config['shifting']['surfaceDetection'] == 'Maronne':
                particleState['fluidL'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState, config)
                particleState['fluidNormals'], particleState['fluidLambdas'] = computeNormalsMaronne(particleState, config)
                particleState['fluidFreeSurface'], cA, cB = detectFreeSurfaceMaronne(particleState, config)
            elif config['shifting']['surfaceDetection'] == 'colorGrad':
                particleState['fluidColor'] = computeColorField(particleState, config)
                particleState['fluidColorGradient'] = computeColorFieldGradient(particleState, config)
                particleState['fluidFreeSurface'] = detectFreeSurfaceColorFieldGradient(particleState, config)
                            
            particleState['fluidFreeSurfaceMask'] = expandFreeSurfaceMask(particleState, config)


        if config['shifting']['scheme'] == 'IPS':
            update, K, J, H, B, convergence, iters, residual = computeShifting(particleState, config, computeRho = config['shifting']['summationDensity'], scheme = config['shifting']['solver'])
            overallStates.append((convergence, iters, residual))
        else:
            update = -deltaPlusShifting(particleState, config)
            
        if config['shifting']['freeSurface']:
            if config['shifting']['normalScheme'] == 'color':
                ones = torch.ones_like(particleState['fluidSupports'])
                colorField = sphOperationFluidState(particleState, (ones, ones), operation = 'interpolate')
                gradColorField = sphOperationFluidState(particleState, (colorField, colorField), operation = 'gradient', gradientMode = 'difference')
                n = torch.nn.functional.normalize(gradColorField, dim = -1)
                particleState['fluidNormals'] = n
            elif config['shifting']['normalScheme'] == 'lambda':
                if 'fluidL' not in particleState:                        
                    particleState['fluidL'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState, config)
                particleState['fluidNormals'], particleState['fluidLambdas'] = computeNormalsMaronne(particleState, config)
                n = torch.nn.functional.normalize(computeLambdaGrad(particleState, config), dim = -1)
                particleState['fluidNormals'] = n
            else:
                n = particleState['fluidNormals']
                
            fs = particleState['fluidFreeSurface']
            fsm = particleState['fluidFreeSurfaceMask']

            if config['shifting']['projectionScheme'] == 'dot':
                result = update + torch.einsum('ij,ij->i', update, n)[:, None] * n
                update[fsm > 0.5] = result[fsm > 0.5] * config['shifting']['surfaceScaling']
                update[fs > 0.5] = 0
            elif config['shifting']['projectionScheme'] == 'mat':
                nMat = torch.einsum('ij, ik -> ijk', particleState['fluidNormals'], particleState['fluidNormals'])
                M = torch.diag_embed(particleState['fluidPositions'].new_ones(particleState['fluidPositions'].shape)) - nMat
                result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                update[fsm > 0.5] = result[fsm > 0.5] * config['shifting']['surfaceScaling']
                # update[fs > 0.5] = 0
            else:
                update[fs > 0.5] = 0
            
        
        spacing = config['particle']['dx']

        print(
            f'Iter: {i}, Residual: {residual}, Threshold {config["shifting"]["threshold"] * spacing}, max Update = {update.max()} Ratio {update.max() / (config["shifting"]["threshold"] * spacing)}')

        update = torch.clamp(update, -config['shifting']['threshold'] * spacing, config['shifting']['threshold'] * spacing)
        print(f'Update: {update.max()} Ratio: {update.max() / (config["shifting"]["threshold"] * spacing)}')
        particleState['fluidPositions'] = particleState['fluidPositions'] - update

        # print(f'J: {J.abs().max()}')
    dx = particleState['fluidPositions'] - initialPositions
    particleState['fluidPositions'] = initialPositions
    particleState['fluidDensities'] = initialDensities

    return dx, overallStates


from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('shifting', 'scheme', str, 'deltaSPH', required = False,export = False, hint = 'Scheme for the density diffusion term (IPS, deltaPlus)'),
        Parameter('shifting', 'R', float, 0.25, required = False,export = False, hint = 'R parameter for the deltaPlus scheme'),
        Parameter('shifting', 'n', float, 4, required = False,export = False, hint = 'n parameter for the deltaPlus scheme'),
        Parameter('shifting', 'CFL', float, 1.5, required = False,export = False, hint = 'CFL parameter for the deltaPlus scheme'),
        Parameter('shifting', 'computeMach', bool, True, required = False,export = False, hint = 'Compute Mach number for the deltaPlus scheme'),
        Parameter('shifting', 'solver', str, 'BiCGStab_wJacobi', required = False, export = False, hint = 'Use BiCGStab for the IPS scheme'),
        Parameter('shifting', 'freeSurface', bool, False, required = False,export = False, hint = 'Use free surface detection for shifting scheme'),
        Parameter('shifting', 'normalScheme', str, 'lambda', required = False,export = False, hint = 'Scheme for normal computation (color, lambda, default)'),
        Parameter('shifting', 'projectionScheme', str, 'mat', required = False,export = False, hint = 'Scheme for normal projection (dot, M)'),
        Parameter('shifting', 'threshold', float, 0.01, required = False,export = False, hint = 'Threshold for the shifting scheme'),
        Parameter('shifting', 'surfaceScaling', float, 0.1, required = False,export = False, hint = 'Scaling for the surface projection'),
        Parameter('shifting', 'maxIterations', int, 1, required = False,export = False, hint = 'Maximum number of iterations for the shifting scheme'),
        Parameter('shifting', 'summationDensity', bool, False, required = False,export = False, hint = 'Use summation density for the shifting scheme'),
        Parameter('shifting', 'useExtendedMask', bool, False, required = False,export = False, hint = 'Use extended mask for the shifting scheme'),
        Parameter('shifting', 'initialization', str, 'zero', required = False,export = False, hint = 'Initialization scheme for the shifting scheme (deltaPlus, random)'),
        Parameter('shifting', 'maxSolveIter', int, 64, required = False,export = False, hint = 'Maximum number of iterations for the linear solver in the shifting scheme'),
        Parameter('shifting', 'surfaceDetection', str, 'Barcasco', required = False,export = False, hint = 'Surface detection scheme for the free surface (Maronne, colorGrad)'),
        Parameter('shifting', 'active', bool, True, required = False,export = False, hint = 'Enable the shifting scheme')
    ]


from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.surfaceDetection import detectFreeSurfaceBarecasco, computeNormalsMaronne, detectFreeSurfaceMaronne, expandFreeSurfaceMask, computeColorField, computeColorFieldGradient, detectFreeSurfaceColorFieldGradient
# from diffSPH.v2.modules.shifting import computeLambdaGrad, deltaPlusShifting, computeShifting, BiCGStab_wJacobi, BiCG, LinearCG, BiCGStab

def computeShifting(particleState, config, computeRho = False, scheme = 'BiCG'):
    numParticles = particleState['fluidPositions'].shape[0]
    if config['shifting']['freeSurface']:  
        if config['shifting']['useExtendedMask']:
            fs = particleState['fluidFreeSurfaceMask']
        else:
            fs = particleState['fluidFreeSurface']
        fsm = particleState['fluidFreeSurfaceMask']

    # particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
    (i,j) = particleState['fluidNeighborhood']['indices']
    rij = particleState['fluidNeighborhood']['distances']
    xij = particleState['fluidNeighborhood']['vectors']
    hij = particleState['fluidNeighborhood']['supports']
    k = config['kernel']['function']
    dim = config['domain']['dim']

    K, J, H = evalKernel(rij, xij, hij, k, dim)
    if computeRho:
        particleState['fluidDensities'] = sphOperationFluidState(particleState, None, 'density')
        omega =  particleState['fluidMasses'] / particleState['fluidDensities']
    else:
        omega = particleState['fluidAreas']
    
    
    J = scatter_sum(J * omega[j,None], i, dim = 0, dim_size = numParticles)
    H = H * omega[j,None,None]

    h2 = particleState['fluidSupports'].repeat(2,1).T.flatten()
    h2 = config['particle']['dx']
    x0 = torch.rand(numParticles * 2).to(rij.device).type(rij.dtype) * h2 / 4 - h2 / 8
    if config['shifting']['initialization'] == 'deltaPlus':
        x0 = -deltaPlusShifting(particleState, config).flatten() * 0.5
    if config['shifting']['initialization'] == 'deltaMinus':
        x0 = deltaPlusShifting(particleState, config).flatten() * 0.5
    if config['shifting']['initialization'] == 'zero':
        x0 = torch.zeros_like(x0)
    

    B = torch.zeros(numParticles * 2, dtype = torch.float32, device=rij.device)
    if config['shifting']['freeSurface']:

        J2 = torch.zeros(J.shape[0], 2, dtype = torch.float32, device=rij.device)
        J2[fs < 0.5, :] = J[fs < 0.5, :]

        B[::2] = J2[:,0]
        B[1::2] = J2[:,1]

        x0 = x0.view(-1,2)
        x0[fs > 0.5,0] = 0
        x0[fs > 0.5,1] = 0
        x0 = x0.flatten()
        H[fs[i] > 0.5,:,:] = 0
        # H[fs[j] > 0.5,:,:] = 0
        iMasked = i[fs[i] < 0.5]
        jMasked = j[fs[i] < 0.5]
        if scheme == 'BiCG':
            xk, convergence, iters, residual = BiCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab':
            xk, convergence, iters, residual = BiCGStab(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab_wJacobi':
            xk, convergence, iters, residual = BiCGStab_wJacobi(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        else:
            xk, convergence, iters, residual = LinearCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
        # print(f'Iter: {iters}, Residual: {residual}, fs: xk fs: {update[fs > 0.5]}')
        # if scheme == 'BiCG':
        #     xk = BiCG(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        # elif scheme == 'BiCGStab':
        #     xk = BiCGStab(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        # elif scheme == 'BiCGStab_wJacobi':
        #     xk = BiCGStab_wJacobi(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        # else:
        #     xk = LinearCG(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
    else:
        B[::2] = J[:,0]
        B[1::2] = J[:,1]
        if scheme == 'BiCG':
            xk, convergence, iters, residual = BiCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab':
            xk, convergence, iters, residual = BiCGStab(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab_wJacobi':
            xk, convergence, iters, residual = BiCGStab_wJacobi(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])
        else:
            xk, convergence, iters, residual = LinearCG(H, B, x0, i, j, maxIter = config['shifting']['maxSolveIter'])

    update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
    return update, K, J, H, B, convergence, iters, residual

def solveShifting(particleState, config):
    initialPositions = torch.clone(particleState['fluidPositions'])
    initialDensities = torch.clone(particleState['fluidDensities'])
    overallStates = []
    for i in range(config['shifting']['maxIterations']):
        particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
        if config['shifting']['summationDensity']:
            particleState['fluidDensities'] = sphOperationFluidState(particleState, None, 'density')

        if config['shifting']['freeSurface']:
            if config['shifting']['surfaceDetection'] == 'Maronne':
                particleState['fluidL'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState, config)
                particleState['fluidNormals'], particleState['fluidLambdas'] = computeNormalsMaronne(particleState, config)
                particleState['fluidFreeSurface'], cA, cB = detectFreeSurfaceMaronne(particleState, config)
            elif config['shifting']['surfaceDetection'] == 'colorGrad':
                particleState['fluidColor'] = computeColorField(particleState, config)
                particleState['fluidColorGradient'] = computeColorFieldGradient(particleState, config)
                particleState['fluidFreeSurface'] = detectFreeSurfaceColorFieldGradient(particleState, config)
            elif config['shifting']['surfaceDetection'] == 'Barcasco':
                particleState['fluidFreeSurface'] = detectFreeSurfaceBarecasco(particleState, config)
                            
            particleState['fluidFreeSurfaceMask'] = expandFreeSurfaceMask(particleState, config)


        if config['shifting']['scheme'] == 'IPS':
            update, K, J, H, B, convergence, iters, residual = computeShifting(particleState, config, computeRho = config['shifting']['summationDensity'], scheme = config['shifting']['solver'])
            overallStates.append((convergence, iters, residual))
        else:
            update = -deltaPlusShifting(particleState, config)
            
        if config['shifting']['freeSurface']:
            if config['shifting']['normalScheme'] == 'color':
                ones = torch.ones_like(particleState['fluidSupports'])
                colorField = sphOperationFluidState(particleState, (ones, ones), operation = 'interpolate')
                gradColorField = sphOperationFluidState(particleState, (colorField, colorField), operation = 'gradient', gradientMode = 'difference')
                n = torch.nn.functional.normalize(gradColorField, dim = -1)
                particleState['fluidNormals'] = n
            elif config['shifting']['normalScheme'] == 'lambda':
                # if 'fluidL' not in particleState:                        
                particleState['fluidL'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState, config)
                particleState['fluidNormals'], particleState['fluidLambdas'] = computeNormalsMaronne(particleState, config)
                n = torch.nn.functional.normalize(computeLambdaGrad(particleState, config), dim = -1)
                particleState['fluidNormals'] = n
            else:
                n = particleState['fluidNormals']
                
            fs = particleState['fluidFreeSurface']
            fsm = particleState['fluidFreeSurfaceMask']
            # print(update[fs > 0.5].abs().max())
            if config['shifting']['projectionScheme'] == 'dot':
                result = update + torch.einsum('ij,ij->i', update, n)[:, None] * n
                update[fsm > 0.5] = result[fsm > 0.5] * config['shifting']['surfaceScaling']
                # update[fs > 0.5] = 0
                update[particleState['fluidLambdas'] < 0.4] = 0
            elif config['shifting']['projectionScheme'] == 'mat':
                nMat = torch.einsum('ij, ik -> ikj', particleState['fluidNormals'], particleState['fluidNormals'])
                M = torch.diag_embed(particleState['fluidPositions'].new_ones(particleState['fluidPositions'].shape)) - nMat
                result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                update[fsm > 0.5] = result[fsm > 0.5] 
                update[particleState['fluidLambdas'] < 0.4] = 0
                update[fs > 0.5] = update[fs > 0.5] * config['shifting']['surfaceScaling']
            else:
                update[particleState['fluidLambdas'] < 0.4] = 0
                # update[fs > 0.5] = 0
            
        
        spacing = config['particle']['dx']
        # print(
        #     f'Iter: {i}, Threshold {config["shifting"]["threshold"] * spacing}, max Update = {update.max()} Ratio {update.max() / (config["shifting"]["threshold"] * spacing)}')

        update = torch.clamp(update, -config['shifting']['threshold'] * spacing, config['shifting']['threshold'] * spacing)
        # print(f'Update: {update.max()} Ratio: {update.max() / (config["shifting"]["threshold"] * spacing)}')
        # update = torch.clamp(update, -config['shifting']['threshold'] * spacing, config['shifting']['threshold'] * spacing)
        particleState['fluidPositions'] = particleState['fluidPositions'] - update

        # print(f'J: {J.abs().max()}')
    dx = particleState['fluidPositions'] - initialPositions
    particleState['fluidPositions'] = initialPositions
    particleState['fluidDensities'] = initialDensities

    return dx, overallStates