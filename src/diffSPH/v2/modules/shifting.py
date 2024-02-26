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

def deltaPlusShifting(particleState, config):
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
        Ma = torch.linalg.norm(particleState['fluidVelocities'], dim = -1) / config['fluid']['cs']
    shiftScaling = -CFL * Ma * (particleState['fluidSupports'] * 2)**2
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

def computeShifting(particleState, config, computeRho = False, BiCG = True):
    numParticles = particleState['fluidPositions'].shape[0]
    if config['shifting']['freeSurface']:  
        if config['shifting']['useExtendedMask']:
            fs = particleState['fluidFreeSurfaceMask']
        else:
            fs = particleState['fluidFreeSurface']
        fsm = particleState['fluidFreeSurfaceMask']

    particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
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
    x0 = torch.rand(numParticles * 2).to(rij.device).type(rij.dtype) * h2 / 4 - h2 / 8

    B = torch.zeros(numParticles * 2, dtype = torch.float32)
    if config['shifting']['freeSurface']:

        J2 = torch.zeros(J.shape[0], 2, dtype = torch.float32)
        J2[fs < 0.5, :] = J[fs < 0.5, :]

        B[::2] = J2[:,0]
        B[1::2] = J2[:,1]

        x0 = x0.view(-1,2)
        x0[fs > 0.5,0] = 0
        x0[fs > 0.5,1] = 0
        x0 = x0.flatten()
            
        iMasked = i[fs[i] < 0.5]
        jMasked = j[fs[i] < 0.5]
        if BiCG:
            xk = BiCGStab(H[fs[i] < 0.5], B, x0, iMasked, jMasked)
        else:
            xk = LinearCG(H[fs[i] < 0.5], B, x0, iMasked, jMasked)
    else:
        B[::2] = J[:,0]
        B[1::2] = J[:,1]
        if BiCG:
            xk = BiCGStab(H, B, x0, i, j)
        else:
            xk = LinearCG(H, B, x0, i, j)

    update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
    return update, K, J, H, B

from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne, expandFreeSurfaceMask

def solveShifting(particleState, config):
    initialPositions = torch.clone(particleState['fluidPositions'])
    initialDensities = torch.clone(particleState['fluidDensities'])

    for i in range(config['shifting']['maxIterations']):
        particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
        if config['shifting']['summationDensity']:
            particleState['fluidDensities'] = sphOperationFluidState(particleState, None, 'density')

        if config['shifting']['freeSurface']:
            particleState['fluidL'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState, config)
            particleState['fluidNormals'], particleState['fluidLambdas'] = computeNormalsMaronne(particleState, config)
            particleState['fluidFreeSurface'], cA, cB = detectFreeSurfaceMaronne(particleState, config)
            particleState['fluidFreeSurfaceMask'] = expandFreeSurfaceMask(particleState, config)

        if config['shifting']['scheme'] == 'IPS':
            update, K, J, H, B = computeShifting(particleState, config, computeRho = config['shifting']['summationDensity'], BiCG = config['shifting']['BiCG'])
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
                n = torch.nn.functional.normalize(computeLambdaGrad(particleState, config), dim = -1)
                particleState['fluidNormals'] = n
            else:
                n = particleState['fluidNormals']
                
            fs = particleState['fluidFreeSurface']
            fsm = particleState['fluidFreeSurfaceMask']

            if config['shifting']['projectionScheme'] == 'dot':
                result = update + torch.einsum('ij,ij->i', update, n)[:, None] * n
                update[fsm > 0.5] = result[fsm > 0.5] * 0.125
                update[fs > 0.5] = 0
            else:
                nMat = torch.einsum('ij, ik -> ijk', particleState['fluidNormals'], particleState['fluidNormals'])
                M = torch.diag_embed(particleState['fluidPositions'].new_ones(particleState['fluidPositions'].shape)) - nMat
                result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                update[fsm > 0.5] = result[fsm > 0.5] * 0.5
                update[fs > 0.5] = 0
        
        spacing = config['particle']['dx']
        update = torch.clamp(update, -config['shifting']['threshold'] * spacing, config['shifting']['threshold'] * spacing)
        particleState['fluidPositions'] = particleState['fluidPositions'] - update

        # print(f'J: {J.abs().max()}')
    dx = particleState['fluidPositions'] - initialPositions
    particleState['fluidPositions'] = initialPositions
    particleState['fluidDensities'] = initialDensities

    return dx


from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('shifting', 'scheme', str, 'IPS', required = False,export = False, hint = 'Scheme for the density diffusion term (IPS, deltaPlus)'),
        Parameter('shifting', 'R', float, 0.25, required = False,export = False, hint = 'R parameter for the deltaPlus scheme'),
        Parameter('shifting', 'n', float, 4, required = False,export = False, hint = 'n parameter for the deltaPlus scheme'),
        Parameter('shifting', 'CFL', float, 1.5, required = False,export = False, hint = 'CFL parameter for the deltaPlus scheme'),
        Parameter('shifting', 'computeMach', bool, False, required = False,export = False, hint = 'Compute Mach number for the deltaPlus scheme'),
        Parameter('shifting', 'BiCG', bool, True, required = False, export = False, hint = 'Use BiCGStab for the IPS scheme'),
        Parameter('shifting', 'freeSurface', bool, False, required = False,export = False, hint = 'Use free surface detection for shifting scheme'),
        Parameter('shifting', 'normalScheme', str, 'lambda', required = False,export = False, hint = 'Scheme for normal computation (color, lambda, default)'),
        Parameter('shifting', 'projectionScheme', str, 'mat', required = False,export = False, hint = 'Scheme for normal projection (dot, M)'),
        Parameter('shifting', 'threshold', float, 0.25, required = False,export = False, hint = 'Threshold for the shifting scheme'),
        Parameter('shifting', 'maxIterations', int, 32, required = False,export = False, hint = 'Maximum number of iterations for the shifting scheme'),
        Parameter('shifting', 'summationDensity', bool, True, required = False,export = False, hint = 'Use summation density for the shifting scheme'),
        Parameter('shifting', 'useExtendedMask', bool, False, required = False,export = False, hint = 'Use extended mask for the shifting scheme'),
    ]