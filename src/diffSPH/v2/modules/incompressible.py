from diffSPH.v2.sparse import _get_atol_rtol, matvec_sparse_coo, make_id
from typing import Tuple, Optional, List
from diffSPH.v2.sphOps import sphOperation, sphOperationStates
from diffSPH.v2.math import scatter_sum

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import scipy.sparse as sp


# @torch.jit.script
def cg_sparse(
    A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int],
    b,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    M: Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]] = None,
    removeMean: bool = False,
):
    # with record_function("cg"):
    if M is None:
        M = make_id(A)
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    bnrm2 = torch.linalg.norm(b)

    atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)

    convergence: List[List[torch.Tensor]] = []

    if bnrm2 == 0:
        return b, 0, convergence

    n = len(b)

    if maxiter is None:
        maxiter = n * 10

    # dotprod = torch.dot

    # matvec = A.matvec
    # psolve = M.matvec
    r = b - matvec_sparse_coo(A, x) if x.any() else b.clone()

    # Dummy value to initialize var, silences warnings
    rho_prev = torch.zeros_like(b)
    p = torch.zeros_like(b)

    for iteration in range(maxiter):
        if torch.linalg.norm(r) < atol:  # Are we done?
            return x, iteration, convergence
        # with record_function("cg_iter"):
        z = matvec_sparse_coo(M, r)
        rho_cur = torch.dot(r, z)
        if iteration > 0:
            # beta = rho_cur / (rho_prev +  + atol * 0.01)
            beta = rho_cur / (rho_prev)
            if torch.any(torch.isnan(beta)):
                print(f'nan in beta: {beta} at iteration {iteration} for {rho_cur} / {rho_prev}')
            if torch.any(torch.isinf(beta)):
                print(f'inf in beta: {beta} at iteration {iteration} for {rho_cur} / {rho_prev}')
            p *= beta
            p += z
        else:  # First spin
            p = z.clone()
            # p = torch.empty_like(r)
            # p[:] = z[:]

        # q = matvec_sparse_coo(A, p)
        q = matvec_sparse_coo(A, p)
        alpha = rho_cur / (torch.dot(p, q))
        # alpha = rho_cur / (torch.dot(p, q)- 1 / p.shape[0] * torch.dot(p,p) + atol * 0.01)
        # alpha = rho_cur / (torch.dot(p, q)- 1 / p.shape[0] * torch.dot(p,p))

        if torch.any(torch.isnan(alpha)):
            print(f'nan in alpha: {alpha} at iteration {iteration} for {rho_cur} / {torch.dot(p, q)}')
        if torch.any(torch.isinf(alpha)):
            print(f'inf in alpha: {alpha} at iteration {iteration} for {rho_cur} / {torch.dot(p, q)}')
        x += alpha * p
        if removeMean:
            x = x - x.mean()
        r -= alpha * q  # - 1 / p.shape[0] * torch.dot(p,p)

        if torch.any(torch.isnan(r)):
            print(f'nan in r at iteration {iteration}')
        if torch.any(torch.isnan(x)):
            print(f'nan in x at iteration {iteration}')

        rho_prev = rho_cur

        # if callback:
        # callback(x)
        convergence.append([x.clone(), torch.linalg.norm(matvec_sparse_coo(A, x) - b)])

    # else:  # for loop exhausted
    # Return incomplete progress
    return x, maxiter, convergence


def computeLaplacian(stateA, stateB, config, neighborhood, dt):
    # stateA = solverState['fluid']
    # stateB = solverState['fluid']
    # neighborhood = stateA['neighborhood']

    i, j = neighborhood['indices']
    rij = neighborhood['distances']
    hij = neighborhood['supports']
    # xij = neighborhood['vectors']
    gradKernels = neighborhood['gradients']

    quotient = (rij * hij + 1e-7 * hij)
    kernelApproximation = torch.linalg.norm(gradKernels, dim=-1) / quotient
    # kernelApproximation = torch.einsum('nd, nd -> n', gradKernels, -xij)/  quotient# * rij * hij

    # Aij = pA[i] - pB[j]
    massTerm = stateA['masses'][j] / (stateB['densities'][j])  # **2
    dimTerm = config['domain']['dim'] * 2 + 2
    dimTerm = 8
    dimTerm = 2

    laplacianTerm = dt * dimTerm * massTerm * kernelApproximation

    aii = scatter_sum(laplacianTerm, i, dim=0, dim_size=stateA['positions'].shape[0])

    mask = i == j

    A_ = laplacianTerm.clone()
    A_[mask] = -aii
    A = (A_, (i, j), stateA['numParticles'])

    return A, aii


def computeDivergence(stateA, stateB, neighborhood, simConfig, key="velocities"):
    with record_function("[SPH] - Fluid Divergence ($nabla cdot \\bar{v}$)"):
        divergence = -sphOperationStates(
            stateA,
            stateB,
            (stateA[key], stateB[key]),
            neighborhood=neighborhood,
            operation="divergence",
            gradientMode = 'difference',
        )
        return divergence


def computeDivergenceSourceTerm(stateA, stateB, config, neighborhood, dt):
    stateA["advectionVelocities"] = stateA["velocities"] + stateA["advection"] * dt

    stateA["advectionDivergence"] = sphOperationStates(
        stateA,
        stateB,
        (stateA["advectionVelocities"], stateA["advectionVelocities"]),
        operation="divergenceConsistent",
        neighborhood=stateA["neighborhood"],
        gradientMode="difference",
    )
    # computeDivergence(stateA, stateB, neighborhood, config, key = 'advectionVelocities') / 10**5

    rho = stateA["densities"]
    rho = config["fluid"]["rho0"] * torch.ones_like(rho)
    B = rho * stateA["advectionDivergence"]  # / dt #config['timestep']['dt'] #* rhoStar
    # print(
    #     f"Divergence-Free Source Term: {B.abs().max()} | {B.abs().min()} | {B.mean()}"
    # )
    B -= B.mean()
    return B#.to(torch.float64)


def projectVelocities(stateA, stateB, config, neighborhood, dt, returnSparseSystem=False, solveOnCPU=False, solverPrecision=torch.float32):
    numParticles = stateA['numParticles']
    device = stateA['positions'].device
    dtype = stateA['positions'].dtype

    A, aii = computeLaplacian(stateA, stateB, config, neighborhood, dt=dt)
    B = computeDivergenceSourceTerm(stateA, stateB, config, neighborhood, dt)

    M = -1 / aii
    M_i = torch.arange(0, stateA['numParticles'], device=stateA['positions'].device)
    M_j = torch.arange(0, stateA['numParticles'], device=stateA['positions'].device)
    M_coo = (M, (M_i, M_j), stateA['numParticles'])
    x0 = torch.zeros_like(B) if 'pressure' not in stateA else stateA['pressure'].clone() * 0.5

    if solveOnCPU:
        A = (A[0].cpu(), (A[1][0].cpu(), A[1][1].cpu()), A[2])
        B = B.cpu()
        M_coo = (M_coo[0].cpu(), (M_coo[1][0].cpu(), M_coo[1][1].cpu()), M_coo[2])
        x0 = x0.cpu()
    if solverPrecision != torch.float32:
        A = (A[0].to(solverPrecision), A[1], A[2])
        B = B.to(solverPrecision)
        M_coo = (M_coo[0].to(solverPrecision), M_coo[1], M_coo[2])
        x0 = x0.to(solverPrecision)

    residuals = []

    p, iters, convergence = cg_sparse(A, B, x0, rtol=1e-2, maxiter=512, M=M_coo, removeMean=True)
    if solveOnCPU or solverPrecision != torch.float32:
        p = p.to(dtype).to(device)

    # print(f'Torch CG: {iters} iterations, {convergence[-1][1] if len(convergence) > 0 else 0} residual, {p.max()} | {p.min()} | {p.mean()}')

    if returnSparseSystem:
        A_cpu = (A[0].cpu(), (A[1][0].cpu(), A[1][1].cpu()), A[2])
        M_coo_cpu = (M_coo[0].cpu(), (M_coo[1][0].cpu(), M_coo[1][1].cpu()), M_coo[2])
        B_cpu = B.cpu()

        A_sp = sp.coo_matrix(
            (
                A_cpu[0].to(torch.float64).detach().numpy(),
                (
                    A_cpu[1][0].detach().numpy(),
                    A_cpu[1][1].detach().cpu().detach().numpy(),
                ),
            ),
            shape=[numParticles, numParticles],
        )
        M_sp = sp.coo_matrix(
            (
                M_coo_cpu[0].detach().to(torch.float64).numpy(),
                (M_coo_cpu[1][0].detach().numpy(), M_coo_cpu[1][1].detach().numpy()),
            ),
            shape=[numParticles, numParticles],
        )
        rhs = B_cpu.detach().to(torch.float64).numpy()

        sparseSystem = (A_sp, M_sp, rhs)
    else:
        sparseSystem = None

    return p, len(convergence), convergence, residuals, A, B, sparseSystem


def computeAlpha(stateA, stateB, config, neighborhood, dt, actualAreas):

    fluidNeighbors = neighborhood  # simulationState['fluid']['neighborhood']
    (i, j) = fluidNeighbors['indices']

    grad = fluidNeighbors['gradients']
    grad2 = torch.einsum('nd, nd -> n', grad, grad)

    term1 = actualAreas[1][j][:,None] * grad
    term2 = (actualAreas[1]**2 / (stateB['areas'] * config['fluid']['rho0']))[j] * grad2

    kSum1 = scatter_sum(term1, i, dim=0, dim_size=stateA['areas'].shape[0])
    kSum2 = scatter_sum(term2, i, dim=0, dim_size=stateA['areas'].shape[0])

    fac = - dt**2 * actualAreas[0]
    mass = stateA['areas'] * config['fluid']['rho0']
    alpha = fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2
    # alpha = torch.clamp(alpha, -1, -1e-7)

    # alpha = alpha * dt
    return alpha / dt


def computePressureAcceleration(stateA, stateB, config, neighborhood, pressure):
    # sphOperationStates(solverState['fluid'], solverState['fluid'], (p_starstar, p_starstar), operation = 'gradient', gradientMode='symmetric', neighborhood= solverState['fluid']['neighborhood'])
    return -sphOperationStates(
        stateA,
        stateB,
        (pressure[0], pressure[1]),
        operation="gradient",
        gradientMode="difference",
        neighborhood=neighborhood,
    ) / stateA["densities"].view(-1, 1) # config['fluid']['rho0'] ## stateA["densities"].view(-1, 1)


def updatePressure(
    stateA,
    stateB,
    config,
    neighborhood,
    dt,
    pressures,
    pressureAccels,
    sourceTerms,
    alphas,
):
    kernelSum = -(dt**2) * sphOperationStates(
        stateA,
        stateB,
        (pressureAccels[0], pressureAccels[1]),
        operation="divergence",
        gradientMode="difference",
        neighborhood=neighborhood,
    )

    sourceTerm = sourceTerms[0]
    residual = kernelSum - sourceTerm
    pressure = pressures[0] + 0.5 * (sourceTerm - kernelSum) / alphas[0]

    # if config['dfsph']['clampPressure']:
    # pressure = torch.max(pressure, torch.zeros_like(pressure))

    return pressure, residual


def solveIncompressible_relaxedJacobi(stateA, stateB, config, dt, neighborhood):
    rho = sphOperationStates(stateA, stateB, quantities=None, operation="density", neighborhood=neighborhood)
    advectionVelocities = stateA["velocities"] + stateA["advection"] * dt

    advectionDivergence = -sphOperationStates(stateA, stateB, (advectionVelocities, advectionVelocities), neighborhood=neighborhood, operation="divergence",gradientMode="difference")

    actualArea = stateA["masses"] / rho
    # direct formulation
    rho_initial = rho
    rhoStar = rho_initial - rho_initial * dt * advectionDivergence

    # sourceTerm = (config['fluid']['rho0'] - rhoStar) / timestep
    # PBSPH formulation
    sourceTerm = 1 - stateA["areas"] / actualArea + dt * advectionDivergence

    pressureA = torch.zeros(stateA["numParticles"], device=config["compute"]["device"])
    pressureB = torch.zeros(stateA["numParticles"], device=config["compute"]["device"])

    if "pressureIncompressible" in stateA:
        pressureB = 0.5 * stateA["pressureIncompressible"].clone()
        pressureA = 0.5 * stateA["pressureIncompressible"].clone()

    # print(f"Initial Density: {rho.abs().max()} | {rho.abs().min()} | {rho.mean()}")
    # print(f"Predicted Density: {rhoStar.abs().max()} | {rhoStar.abs().min()} | {rhoStar.mean()}")
    # print(f"Source Term: {sourceTerm.abs().max()} | {sourceTerm.abs().min()} | {sourceTerm.mean()}")

    errors = []
    pressures = []
    i = 0
    error = 0.0
    minIters = 2  # config['dfsph']['minIters']
    maxIters = 256  # config['dfsph']['maxIters']
    errorThreshold = 5e-4  # config['dfsph']['errorThreshold']
    # errorThreshold = 1e-5

    alpha = computeAlpha(stateA, stateB, config, neighborhood, dt=dt, actualAreas=(actualArea, actualArea))  # / fac

    while i < maxIters and (i < minIters or error > errorThreshold):
        pressureAccel = computePressureAcceleration(
            stateA, stateB, config, neighborhood, (pressureB, pressureB)
        )
        pressureA[:] = pressureB.clone()

        pressureB, residual = updatePressure( stateA, stateB, config, neighborhood, dt=dt, pressures=(pressureA, pressureB), pressureAccels=(pressureAccel, pressureAccel), sourceTerms=(sourceTerm, sourceTerm), alphas=(alpha, alpha))
        # if config['dfsph']['sourceTerm'] == 'density':
        # stateA['pressureB'] = torch.max(stateA['pressureB'], torch.zeros_like(stateA['pressureB']))
        # error = torch.mean(torch.clamp(stateA['residual'], min = -errorThreshold))

        error = torch.mean(torch.clamp(residual.abs(), min=-errorThreshold))  # / config['fluid']['rho0']

        errors.append(error.detach().cpu().item())
        # print(f"{i:2d} -> {error.detach().cpu().item():+.4e}, pressure mean: {pressureB.mean().detach().cpu().item():+.4e}, pressure accel mean: {torch.linalg.norm(pressureAccel, dim=-1).mean().detach().cpu().item():+.4e}")
        # break
        pressures.append(pressureB.mean().detach().cpu().item())

        i += 1
    return pressureB, i, errors, pressures, sourceTerm

from diffSPH.v2.modules.neighborhood import searchNeighbors
from diffSPH.v2.simulationSchemes.dfsph import callModule, computeDensity, computeGravity, computeViscosity, sphOperationStates, scatter_sum

def simulationStep(solverState, config):
    searchNeighbors(solverState, config)

    # Advection Step
    solverState['fluid']['densities'], _ = callModule(solverState, computeDensity, config, 'fluid')
    solverState['fluid']['gravityAccel'] = computeGravity(solverState['fluid'], config)
    solverState['fluid']['velocityDiffusion'], _ = callModule(solverState, computeViscosity, config, 'fluid')

    # Divergence-Free Solver Preparation
    solverState['fluid']['advection'] = solverState['fluid']['gravityAccel'] + solverState['fluid']['velocityDiffusion']
    solverState['fluid']['initialVelocities'] = solverState['fluid']['velocities'].clone()
    solverState['fluid']['advectionVelocities'] = solverState['fluid']['velocities'] + solverState['fluid']['advection'] * solverState['dt']
    solverState['fluid']['velocityDivergence'] = computeDivergence(solverState['fluid'], solverState['fluid'], solverState['fluid']['neighborhood'], config)
    solverState['fluid']['advectionDivergence'] = computeDivergence(solverState['fluid'], solverState['fluid'], solverState['fluid']['neighborhood'], config, key='advectionVelocities')

    v_star = solverState['fluid']['velocities'] + solverState['fluid']['advection'] * solverState['dt']
    # Could also use actual density here
    rho = torch.ones_like(solverState['fluid']['densities']) * config['fluid']['rho0']  # solverState['fluid']['densities'].view(-1,1)

    # Solve Divergence-Free PPE
    pressure_star, iters, convergence, residuals, A, B, (A_sp, M_coo, rhs) = projectVelocities(solverState['fluid'], solverState['fluid'], config, solverState['fluid']['neighborhood'], solverState['dt'], returnSparseSystem=True)
    iters = len(convergence)
    # print(f'Divergence-free convergence after {iters} iterations: {convergence[-1][1] if len(convergence) > 0 else 0}')
    # print(f'Pressure Stats: {pressure_star.max()} | {pressure_star.min()} | {pressure_star.mean()}')

    gradp_star = sphOperationStates(solverState['fluid'], solverState['fluid'], (pressure_star, pressure_star), operation='gradient', gradientMode='summation', neighborhood=solverState['fluid']['neighborhood'])
    accel_p_star = -gradp_star / rho.view(-1, 1)
    # print(f'Pressure Acceleration: {accel_p_star.max()} | {accel_p_star.min()} | {accel_p_star.mean()}')
    solverState['fluid']['pressure'] = pressure_star
    solverState['fluid']['pressureAcceleration'] = accel_p_star
          
    ## Check for proper convergence behavior
    initialDivergence = sphOperationStates(solverState['fluid'], solverState['fluid'], (solverState['fluid']['velocities'], solverState['fluid']['velocities']), operation='divergenceConsistent', neighborhood=solverState['fluid']['neighborhood'], gradientMode='difference')
    advectionDivergence = sphOperationStates(solverState['fluid'], solverState['fluid'], (v_star, v_star), operation='divergenceConsistent', neighborhood=solverState['fluid']['neighborhood'])

    # print(f'Initial Divergence : {initialDivergence.min()} | {initialDivergence.max()} | {initialDivergence.mean()}')
    # print(f'Divergence after DF: {advectionDivergence.min()} | {advectionDivergence.max()} | {advectionDivergence.mean()}')
    v_prime = v_star + accel_p_star * solverState['dt']  # / 10**3 
    finalVelocities = v_prime
    finalPositions = solverState['fluid']['positions'] + v_prime * solverState['dt']

    solverState['fluid']['divergenceFreeVelocities'] = v_prime

    ## Prepare incompressible PPE
    v_prime = v_star + accel_p_star * solverState['dt']  # / 10**3 
    solvedDivergence = sphOperationStates(solverState['fluid'], solverState['fluid'], (v_prime, v_prime), operation='divergenceConsistent', neighborhood=solverState['fluid']['neighborhood'])

    ## Solve Incompressible PPE
    # for i in range(1):
        # searchNeighbors(solverState, config)
        # solverState['fluid']['densities'], _ = callModule(solverState, computeDensity, config, 'fluid')
        # if i > 0:
        #     solverState['fluid']['advection'][:,0] = 0
        #     solverState['fluid']['advection'][:,1] = 0
        #     v_prime = finalVelocities

    p_starstar, iters_inc, errors, pressures, sourceTerm = solveIncompressible_relaxedJacobi(solverState['fluid'], solverState['fluid'], config, solverState['dt'], neighborhood=solverState['fluid']['neighborhood'])
    # print(f'Pressure Stats: {p_starstar.max()} | {p_starstar.min()} | {p_starstar.mean()}')
    solverState['fluid']['pressureIncompressible'] = p_starstar

    gradp_starstar = sphOperationStates(solverState['fluid'], solverState['fluid'], (p_starstar, p_starstar), operation='gradient', gradientMode='symmetric', neighborhood= solverState['fluid']['neighborhood']) 
    particleShift = -solverState['dt'] ** 2  * gradp_starstar / rho.view(-1, 1)
    gradV = sphOperationStates(solverState['fluid'], solverState['fluid'], (v_prime, v_prime), operation='gradient', gradientMode='difference', neighborhood=solverState['fluid']['neighborhood'])
    # print(f'particleShift: {particleShift.max()} | {particleShift.min()} | {particleShift.mean()}')

    # Project velocities
    projectedVelocity = v_prime + torch.bmm(gradV, particleShift.unsqueeze(-1)).squeeze(-1)
    finalVelocities = projectedVelocity
    # finalVelocities = v_prime
    # if i == 0:
    
    finalPositions = solverState['fluid']['positions'] + v_prime * solverState['dt'] + particleShift
    # else:
        # finalPositions += particleShift
    solverState['fluid']['particleShift'] = particleShift
    solverState['fluid']['projectedVelocities'] = projectedVelocity
    # print(solverState['dt'])
    solverState['fluid']['velocities'] = finalVelocities
    # solverState['fluid']['positions'] += finalVelocities * solverState['dt']#+ a_pi* solverState['dt']**2
    solverState['fluid']['positions'] = finalPositions

    finalDivergence = sphOperationStates(solverState['fluid'], solverState['fluid'], (finalVelocities, finalVelocities), neighborhood=solverState['fluid']['neighborhood'], operation='divergence')

    frameStats = {
        'initialDivergence': initialDivergence,
        'advectionDivergence': advectionDivergence,
        'solvedDivergence': solvedDivergence,
        'finalDivergence': finalDivergence,

        'divergenceFreeIters': iters,
        'divergenceFreeConvgerence': convergence,
        'divergenceFreeResiduals': residuals,
        'divergenceFreeSourceTerm': B,
        
        'incompressibleIters': iters_inc,
        'incompressibleConvergence': errors,
        'incompressiblePressures': pressures,
        'incompressibleSourceTerm': sourceTerm
    }

    return solverState, frameStats