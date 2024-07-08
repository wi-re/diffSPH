import torch 



from typing import Optional, Tuple
from diffSPH.v2.math import scatter_sum

@torch.jit.script
def matvec_sparse_coo(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], x):
    return scatter_sum(A[0] * x[A[1][1]], A[1][0], dim = 0, dim_size = x.shape[0]) 

@torch.jit.script
def rmatvec_sparse_coo(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], x):
    return scatter_sum(A[0] * x[A[1][0]], A[1][1], dim = 0, dim_size = x.shape[0]) 

@torch.jit.script
def make_id(A : Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]):
    M_precond = A[0].new_ones(A[2])
    M_i = torch.arange(A[2]).to(M_precond.device).to(torch.int64)
    M_j = torch.arange(A[2]).to(M_precond.device).to(torch.int64)

    return (M_precond, (M_i, M_j), A[2])

@torch.jit.script
def _get_atol_rtol(name:str, b_norm:float, atol:float=0., rtol:float=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    # if atol == 'legacy' or atol is None or atol < 0:
    #     msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
    #            "if set, `atol` must be a real, non-negative number.")
    #     raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol
@torch.jit.script
def _get_tensor_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).eps,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")
    
# @torch.jit.script
def bicg(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], b, x0:Optional[torch.Tensor]=None, rtol:float=1e-5, atol:float=0., maxiter:Optional[int]=None, M:Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],int]]=None, maxIter: int = 512, tol: float = 1e-5):
    # A, M, x, b, postprocess = make_system(A, M, x0, b)
    if M is None:
        M = make_id(A)
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    bnrm2 = torch.linalg.norm(b)

    atol, _ = _get_atol_rtol('bicg', bnrm2, atol, rtol)

    convergence = []
    
    if bnrm2 == 0:
        # print('bnrm2 == 0')
        return b, 0, convergence
        # return postprocess(b), 0

    n = len(b)
    dotprod = torch.dot

    if maxiter is None:
        maxiter = n*10

    # matvec, rmatvec = A.matvec, A.rmatvec
    # psolve, rpsolve = M.matvec, M.rmatvec
    # print(torch.finfo(x.dtype).eps**2)
    rhotol = _get_tensor_eps(x)**2
    # print(rhotol)

    # # Dummy values to initialize vars, silence linter warnings
    # rho_prev, p, ptilde = r.clone(), r.clone(), r.clone()
    rho_prev = torch.zeros_like(b)
    p = torch.zeros_like(b)
    ptilde = torch.zeros_like(b)

    r = b - matvec_sparse_coo(A, x) if x.any() else b.clone()

    # return r, 0 , convergence
    rtilde = r.clone()

    for iteration in range(maxiter):
        rNorm = torch.linalg.norm(r)
        if rNorm < atol:  # Are we done?
            return x, iteration , convergence

        # print(f'Iter {iteration}, Residual: {torch.linalg.norm(r)}')

        z = matvec_sparse_coo(M, r)
        ztilde = rmatvec_sparse_coo(M, rtilde)
        # order matters in this dot product
        rho_cur = dotprod(rtilde, z)

        if torch.abs(rho_cur) < rhotol:  # Breakdown case
            return x, -10, convergence

        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
            ptilde *= beta.conj()
            ptilde += ztilde
        else:  # First spin
            p = z.clone()
            ptilde = ztilde.clone()

        q = matvec_sparse_coo(A, p)
        qtilde = rmatvec_sparse_coo(A, ptilde)
        rv = dotprod(ptilde, q)

        if rv == 0:
            return x, -11, convergence

        alpha = rho_cur / rv
        x += alpha*p
        r -= alpha*q
        rtilde -= alpha.conj()*qtilde
        rho_prev = rho_cur

        # if callback:
            # callback(x)
        convergence.append(torch.linalg.norm(matvec_sparse_coo(A, x) - b))


    # else:  # for loop exhausted
        # Return incomplete progress
    return x, maxiter, convergence

# @torch.jit.script
def bicgstab(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], b, x0:Optional[torch.Tensor]=None, tol:float = 1e-5, rtol:float=1e-5, atol:float=0., maxiter:Optional[int]=None, M:Optional[ Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]]=None):
    if M is None:
        M = make_id(A)
    x = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)

    atol, _ = _get_atol_rtol('bicgstab', bnrm2, atol, rtol)
    convergence = []

    if bnrm2 == 0:
        return b, 0, convergence

    n = len(b)

    dotprod = torch.dot

    if maxiter is None:
        maxiter = n*10

    # matvec = A.matvec
    # psolve = M.matvec

    # These values make no sense but coming from original Fortran code
    # sqrt might have been meant instead.
    rhotol = _get_tensor_eps(x)**2
    omegatol = rhotol

    # Dummy values to initialize vars, silence linter warnings
    rho_prev = torch.zeros_like(b)
    omega = 0.
    alpha = 0.
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    # rho_prev, omega, alpha, p, v = None, None, None, None, None

    r = b -  matvec_sparse_coo(A, x) if x.any() else b.clone()
    rtilde = r.clone()

    for iteration in range(maxiter):
        if torch.linalg.norm(r) < atol:  # Are we done?
            return x, iteration, convergence

        rho = dotprod(rtilde, r)
        if torch.abs(rho) < rhotol:  # rho breakdown
            return x, -10, convergence

        if iteration > 0:
            if torch.abs(omega) < omegatol:  # omega breakdown
                return x, -11, convergence

            beta = (rho / rho_prev) * (alpha / omega)
            p -= omega*v
            p *= beta
            p += r
        else:  # First spin
            s = torch.empty_like(r)
            p = r.clone()

        phat = matvec_sparse_coo(M, p)
        v = matvec_sparse_coo(A, phat)
        # print(v)
        rv = dotprod(rtilde, v)
        if rv == 0:
            return x, -11, convergence
        alpha = rho / rv
        r -= alpha*v
        s[:] = r[:]

        if torch.linalg.norm(s) < atol:
            x += alpha*phat
            return x, 0, convergence

        shat = matvec_sparse_coo(M, s)
        t = matvec_sparse_coo(A, shat)
        omega = dotprod(t, s) / dotprod(t, t)
        x += alpha*phat
        x += omega*shat
        r -= omega*t
        rho_prev = rho

        # print(omega, alpha)
        convergence.append(torch.linalg.norm(matvec_sparse_coo(A, x) - b))

    # else:  # for loop exhausted
        # Return incomplete progress
    return x, maxiter, convergence

from torch.profiler import profile, record_function, ProfilerActivity
@torch.jit.script
def cg(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], b, x0:Optional[torch.Tensor]=None, tol :float = 1e-5, rtol :float =1e-5, atol :float =0., maxiter: Optional[int]=None, M: Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]]=None):
    # with record_function("cg"):
    
    if M is None:
        M = make_id(A)
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    bnrm2 = torch.linalg.norm(b)

    atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)

    convergence = []

    if bnrm2 == 0:
        return b, 0, convergence

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = torch.dot

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
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = z.clone()
            # p = torch.empty_like(r)
            # p[:] = z[:]

        q = matvec_sparse_coo(A, p)
        alpha = rho_cur / torch.dot(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        # if callback:
            # callback(x)
        convergence.append(torch.linalg.norm(matvec_sparse_coo(A, x) - b))

    # else:  # for loop exhausted
        # Return incomplete progress
    return x, maxiter, convergence