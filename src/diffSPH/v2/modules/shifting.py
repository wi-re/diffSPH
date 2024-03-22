import torch
from diffSPH.v2.math import scatter_sum
# from diffSPH.v2.neighborhood import neighborSearch
from diffSPH.v2.sphOps import sphOperation
from torch.profiler import record_function

def evalKernel(rij, xij, hij, k, dim):
    with record_function("[Shifting] - Eval Kernel System"):
        K = k.kernel(rij, hij, dim)
        J = k.Jacobi(rij, xij, hij, dim)
        H = k.Hessian2D(rij, xij, hij, dim)

        return K, J, H

@torch.jit.script
def LinearCG(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):    
    with record_function("[Shifting] - Linear CG Solver"):
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
    with record_function("[Shifting] - BiCGStab Solver"):
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
    with record_function("[Shifting] - Delta+ Shifting"):
        W_0 = config['kernel']['function'].kernel(torch.tensor(config['particle']['dx'] / config['particle']['support'] * config['kernel']['kernelScale']), torch.tensor(config['particle']['support']), dim = config['domain']['dim'])
        W_0 = config['kernel']['function'].kernel(torch.tensor(config['particle']['dx'] / config['particle']['support']), torch.tensor(config['particle']['support']), dim = config['domain']['dim'])

        (i,j) = particleState['neighborhood']['indices']
        k = particleState['neighborhood']['kernels'] / W_0
        gradK = particleState['neighborhood']['gradients']

        # print(f'Kernels: {k.shape}, mean: {k.mean()}, gradK: {gradK.shape}, mean: {gradK.mean()}')

        R = config['shifting']['R']
        n = config['shifting']['n']
        term = (1 + R * torch.pow(k, n))
        densityTerm = particleState['masses'][j] / (particleState['densities'][i] + particleState['densities'][j])
        phi_ij = 1

        scalarTerm = term * densityTerm * phi_ij
        shiftAmount = scatter_sum(scalarTerm.view(-1,1) * gradK, i, dim = 0, dim_size = particleState['positions'].shape[0])

        CFL = config['shifting']['CFL']
        if config['shifting']['computeMach'] == False:
            Ma = 0.1
        else:
            Ma = torch.amax(torch.linalg.norm(particleState['velocities'], dim = -1)) / config['fluid']['cs']
        shiftScaling = -CFL * Ma * (particleState['supports'] / config['kernel']['kernelScale'] * 2)**2
        # print(f'Shift: {shiftAmount.abs().max()}, Scaling: {shiftScaling.shape}')
        # print(particleState['fluidSupports'])
        return shiftScaling.view(-1,1) * shiftAmount



def computeLambdaGrad(simulationState, config):    
    with record_function("[Shifting] - Lambda Gradient"):
        (i, j) = simulationState['neighborhood']['indices']

        gradKernel = simulationState['neighborhood']['gradients']
        Ls = simulationState['L'][i]

        normalizedGradients = torch.einsum('ijk,ik->ij', Ls, gradKernel)

        return torch.nn.functional.normalize(sphOperation(
            (simulationState['masses'], simulationState['masses']), 
            (simulationState['densities'], simulationState['densities']),
            (simulationState['Lambdas'], simulationState['Lambdas']),
            simulationState['neighborhood']['indices'], 
            simulationState['neighborhood']['kernels'], normalizedGradients,
            simulationState['neighborhood']['distances'], simulationState['neighborhood']['vectors'], simulationState['neighborhood']['supports'], 
            simulationState['numParticles'], 
            operation = 'gradient', gradientMode = 'difference', divergenceMode = 'div', 
            kernelLaplacians = simulationState['neighborhood']['laplacians'] if 'laplacians' in simulationState['neighborhood'] else None), dim = -1)

from diffSPH.v2.sphOps import sphOperationStates
from diffSPH.v2.modules.neighborhood import neighborSearch

@torch.jit.script
def BiCG(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):
    with record_function("[Shifting] - BiCG Solver"):
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
    with record_function("[Shifting] - BiCGStab Solver w/ Jacobi Preconditioner"):
        xk = x0
        rk = torch.zeros_like(x0)
        numParticles = rk.shape[0] // 2
        ii = torch.unique(i)
        # Calculate the Jacobi preconditioner
        diag = torch.zeros_like(B).view(-1, 2)
        diag[ii,0] = H[i == j, 0, 0]
        diag[ii,1] = H[i == j, 1, 1]
        diag = diag.flatten()

        # diag = torch.vstack((H[i == j, 0, 0], H[i == j, 1, 1])).flatten()
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


from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from diffSPH.v2.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne, expandFreeSurfaceMask, computeColorField, computeColorFieldGradient, detectFreeSurfaceColorFieldGradient

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
    with record_function("[Shifting] - Implicit Particle Shifting (IPS)"):
        numParticles = particleState['numParticles']

        if config['shifting']['freeSurface']:  
            if config['shifting']['useExtendedMask']:
                fs = particleState['freeSurfaceMask']
            else:
                fs = particleState['freeSurface']
            fsm = particleState['freeSurfaceMask']

        # particleState['fluidNeighborhood'] = fluidNeighborSearch(particleState, config)
        (i,j) = particleState['neighborhood']['indices']
        rij = particleState['neighborhood']['distances']
        xij = particleState['neighborhood']['vectors']
        hij = particleState['neighborhood']['supports']
        k = config['kernel']['function']
        dim = config['domain']['dim']

        K, J, H = evalKernel(rij, xij, hij, k, dim)
        if computeRho:
            particleState['densities'] = sphOperationStates(particleState, particleState, None, operation = 'density', neighborhood=particleState['neighborhood'])
            omega =  particleState['masses'] / particleState['densities']
        else:
            omega = particleState['areas']
        
        
        J = scatter_sum(J * omega[j,None], i, dim = 0, dim_size = numParticles)
        H = H * omega[j,None,None]

        h2 = particleState['supports'].repeat(2,1).T.flatten()
        h2 = config['particle']['dx']
        x0 = torch.rand(numParticles * 2).to(rij.device).type(rij.dtype) * h2 / 4 - h2 / 8
        if config['shifting']['initialization'] == 'deltaPlus':
            x0 = -deltaPlusShifting(particleState, config).flatten() * 0.5
        if config['shifting']['initialization'] == 'deltaMinus':
            x0 = deltaPlusShifting(particleState, config).flatten() * 0.5
        if config['shifting']['initialization'] == 'zero':
            x0 = torch.zeros_like(x0)
        

        B = torch.zeros(numParticles * 2, dtype = torch.float32, device=rij.device)

        # iActual = i
        # jActual = j
        activeMask = torch.ones_like(i, dtype = torch.bool)
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
            activeMask = fs[i] < 0.5
            # iActual = i[fs[i] < 0.5]
            # jActual = j[fs[i] < 0.5]

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

        if torch.any(particleState['boundaryMarker'] > 0):
            # J[particleState['boundaryMarker'] > 0, :] = 0
            B.view(-1,2)[particleState['boundaryMarker'] > 0, 0] = 0
            H[particleState['boundaryMarker'][i] > 0, :, :] = 0
            x0.view(-1,2)[particleState['boundaryMarker'] > 0, 0] = 0
            if config['shifting']['freeSurface']:
                activeMask = torch.logical_and(particleState['boundaryMarker'][i] == 0, fs[i] < 0.5)
            else:
                activeMask = particleState['boundaryMarker'][i] == 0

        if scheme == 'BiCG':
            xk, convergence, iters, residual = BiCG(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab':
            xk, convergence, iters, residual = BiCGStab(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        elif scheme == 'BiCGStab_wJacobi':
            xk, convergence, iters, residual = BiCGStab_wJacobi(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        else:
            xk, convergence, iters, residual = LinearCG(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])

        update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
        return update, K, J, H, B, convergence, iters, residual

def solveShifting(simulationState, config):
    with record_function("[Shifting] - Compute Shift Amount"):
        numParticles = simulationState['fluid']['numParticles']
        fluidState = simulationState['fluid']
        boundaryParticleState = simulationState['boundary'] if 'boundary' in simulationState else None

        mergedPositions = torch.cat((fluidState['positions'], boundaryParticleState['positions']), dim = 0) if boundaryParticleState is not None else fluidState['positions']
        mergedVelocities = torch.cat((fluidState['velocities'], boundaryParticleState['velocities']), dim = 0) if boundaryParticleState is not None else fluidState['velocities']
        mergedMasses = torch.cat((fluidState['masses'], boundaryParticleState['masses']), dim = 0) if boundaryParticleState is not None else fluidState['masses']
        mergedAreas = torch.cat((fluidState['areas'], boundaryParticleState['areas']), dim = 0) if boundaryParticleState is not None else fluidState['areas']
        mergedDensities = torch.cat((fluidState['densities'], boundaryParticleState['densities']), dim = 0) if boundaryParticleState is not None else fluidState['densities']
        mergedSupports = torch.cat((fluidState['supports'], boundaryParticleState['supports']), dim = 0) if boundaryParticleState is not None else fluidState['supports']
        boundaryMarker = torch.cat((torch.zeros(fluidState['positions'].shape[0], dtype = torch.int32), torch.ones(boundaryParticleState['positions'].shape[0], dtype = torch.int32))) if boundaryParticleState is not None else torch.zeros(fluidState['positions'].shape[0], dtype = torch.int32)
        
        particleState = {
            'positions': mergedPositions,
            'velocities': mergedVelocities,
            'masses': mergedMasses,
            'densities': mergedDensities,
            'supports': mergedSupports,
            'areas': mergedAreas,
            'numParticles': mergedPositions.shape[0],
            'boundaryMarker': boundaryMarker
        }


        initialPositions = torch.clone(particleState['positions'])
        initialDensities = torch.clone(particleState['densities'])
        overallStates = []
        for i in range(config['shifting']['maxIterations']):
            with record_function("[Shifting] - Shift Iteration [Iteration: %3d]" % i):
                with record_function("[Shifting] - Shift Iteration [1 - Neighbor Search]"):
                    particleState['neighborhood'] = neighborSearch(particleState, particleState, config, priorNeighborhood=particleState['neighborhood'] if 'neighborhood' in particleState else None)
                if config['shifting']['summationDensity']:
                    particleState['densities'] = sphOperationStates(particleState, particleState, None, operation = 'density', neighborhood=particleState['neighborhood'])

                with record_function("[Shifting] - Shift Iteration [2 - Surface Detection]"):
                    if config['shifting']['freeSurface']:
                        if config['shifting']['surfaceDetection'] == 'Maronne':
                            particleState['L'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState, particleState, particleState['neighborhood'], config)
                            particleState['normals'], particleState['Lambdas'] = computeNormalsMaronne(particleState, particleState, particleState['neighborhood'], config)
                            particleState['freeSurface'], cA, cB = detectFreeSurfaceMaronne(particleState, particleState, particleState['neighborhood'], config)
                        elif config['shifting']['surfaceDetection'] == 'colorGrad':
                            particleState['color'] = computeColorField(particleState, particleState, particleState['neighborhood'], config)
                            particleState['colorGradient'] = computeColorFieldGradient(particleState, particleState, particleState['neighborhood'], config)
                            particleState['freeSurface'] = detectFreeSurfaceColorFieldGradient(particleState, particleState, particleState['neighborhood'], config)
                        elif config['shifting']['surfaceDetection'] == 'Barcasco':
                            particleState['freeSurface'] = detectFreeSurfaceBarecasco(particleState, particleState, particleState['neighborhood'], config)
                                        
                        particleState['freeSurfaceMask'] = expandFreeSurfaceMask(particleState, particleState, particleState['neighborhood'], config)

                
                with record_function("[Shifting] - Shift Iteration [3 - Shift Computation]"):
                    if config['shifting']['scheme'] == 'IPS':
                        update, K, J, H, B, convergence, iters, residual = computeShifting(particleState, config, computeRho = config['shifting']['summationDensity'], scheme = config['shifting']['solver'])
                        overallStates.append((convergence, iters, residual))
                    else:
                        update = -deltaPlusShifting(particleState, config)
                        # print(f'Update: {update.max()}, {update.min()}')
                with record_function("[Shifting] - Shift Iteration [4 - Surface Projection]"):
                    if config['shifting']['freeSurface']:
                        if config['shifting']['normalScheme'] == 'color':
                            ones = torch.ones_like(particleState['supports'])
                            colorField = sphOperationStates(particleState, particleState, neighborhood = particleState['neighborhood'], quantities = (ones, ones), operation = 'interpolate')
                            gradColorField = sphOperationStates(particleState, particleState, neighborhood = particleState['neighborhood'], quantities = (colorField, colorField), operation = 'gradient', gradientMode = 'difference')
                            n = torch.nn.functional.normalize(gradColorField, dim = -1)
                            particleState['normals'] = n
                        elif config['shifting']['normalScheme'] == 'lambda':
                            # if 'fluidL' not in particleState:                        
                            particleState['L'], normalizationMatrices, particleState['L.EVs'] = computeNormalizationMatrices(particleState,particleState,particleState['neighborhood'], config)
                            particleState['normals'], particleState['Lambdas'] = computeNormalsMaronne(particleState, particleState,particleState['neighborhood'], config)
                            n = torch.nn.functional.normalize(computeLambdaGrad(particleState, config), dim = -1)
                            particleState['normals'] = n
                        else:
                            n = particleState['normals']
                            
                        fs = particleState['freeSurface']
                        fsm = particleState['freeSurfaceMask']
                        # print(update[fs > 0.5].abs().max())
                        if config['shifting']['projectionScheme'] == 'dot':
                            result = update + torch.einsum('ij,ij->i', update, n)[:, None] * n
                            update[fsm > 0.5] = result[fsm > 0.5] * config['shifting']['surfaceScaling']
                            # update[fs > 0.5] = 0
                            update[particleState['Lambdas'] < 0.4] = 0
                        elif config['shifting']['projectionScheme'] == 'mat':
                            nMat = torch.einsum('ij, ik -> ikj', particleState['normals'], particleState['normals'])
                            M = torch.diag_embed(particleState['positions'].new_ones(particleState['positions'].shape)) - nMat
                            result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                            update[fsm > 0.5] = result[fsm > 0.5] 
                            update[particleState['Lambdas'] < 0.4] = 0
                            update[fs > 0.5] = update[fs > 0.5] * config['shifting']['surfaceScaling']
                        else:
                            update[particleState['Lambdas'] < 0.4] = 0
                            # update[fs > 0.5] = 0
                        
                with record_function("[Shifting] - Shift Iteration [5 - Update]"):
                    spacing = config['particle']['dx']
                    # print(
                    #     f'Iter: {i}, Threshold {config["shifting"]["threshold"] * spacing}, max Update = {update.max()} Ratio {update.max() / (config["shifting"]["threshold"] * spacing)}')

                    update = torch.clamp(update, -config['shifting']['threshold'] * spacing, config['shifting']['threshold'] * spacing)
                    update[boundaryMarker != 0,:] = 0
                    # print(f'Update: {update.max()} Ratio: {update.max() / (config["shifting"]["threshold"] * spacing)}')
                    # update = torch.clamp(update, -config['shifting']['threshold'] * spacing, config['shifting']['threshold'] * spacing)
                    particleState['positions'] = particleState['positions'] - update

                # print(f'J: {J.abs().max()}')
        dx = particleState['positions'] - initialPositions
        particleState['positions'] = initialPositions
        particleState['densities'] = initialDensities

        simulationState['fluid']['positions'] = particleState['positions'][:numParticles]
        simulationState['fluid']['densities'] = particleState['densities'][:numParticles]
        dx = dx[:numParticles]

        return dx, overallStates