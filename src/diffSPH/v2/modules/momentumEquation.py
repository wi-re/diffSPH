import torch
from diffSPH.v2.sphOps import sphOperationStates
from torch.profiler import record_function
from diffSPH.v2.math import scatter_sum

def computeMomentumEquation(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Momentum (rho_i nabla cdot v)"):

        # div = sphDivergence((stateA['masses'].to(torch.float64), stateB['masses'].to(torch.float64)), (stateA['densities'].to(torch.float64), stateB['densities'].to(torch.float64)), (stateA['velocities'].to(torch.float64), stateB['velocities'].to(torch.float64)), neighborhood, config['kernel']['gradKernels'].to(torch.float64), stateA['numParticles'], type = 'difference', mode = 'div')

        # return -(stateA['densities'] * div).to(torch.float32)# sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), operation = 'divergence', gradientMode='difference', neighborhood=neighborhood)
        numParticles = stateA['numParticles']
        i,j = neighborhood['indices']
        u_i = stateA['velocities'][i]#.to(torch.float64)
        u_j = stateB['velocities'][j]#.to(torch.float64)

        rho_j = stateB['densities'][j]

        q_ij = (u_j - u_i) * stateB['masses'][j].view(-1,1)
        # q_ij = (u_j - u_i) * stateA['masses'][j].to(torch.float64).view(-1,1)

        k = neighborhood['gradients']#.to(torch.float64)
        kq = torch.einsum('n...d, nd -> n...', q_ij, k)

        kqa = q_ij[:,0] * k[:,0]
        kqb = q_ij[:,1] * k[:,1]
        kq = kqa + kqb        

        kq = kq * stateA['densities'][i] / stateB['densities'][j]
        div = -scatter_sum(kq, i, dim = 0, dim_size = numParticles)#.to(torch.float32)

        # print(f'q_ij: {q_ij.sum()} [{q_ij[:,0].sum()}, {q_ij[:,1].sum()}], k: {k.sum()} [{k[:,0].sum()}, {k[:,1].sum()}], kq: {kq.sum()} [{kqa.sum()}, {kqb.sum()}], div: {div.sum()}')

        # test_q_ij = (u_i - u_j) * stateA['masses'][j].to(torch.float64).view(-1,1)
        # test_kq = torch.einsum('n...d, nd -> n...', test_q_ij, -k)

        # print(torch.all(kq == test_kq))

        return div
        return -sphOperationStates(stateA, stateB, 
            q_ij, operation = 'divergence', gradientMode='difference', neighborhood=neighborhood)
