import torch
from diffSPH.v2.sphOps import sphOperationStates
from torch.profiler import record_function

def computeMomentumEquation(stateA, stateB, neighborhood, config):
    with record_function("[SPH] - Fluid Momentum (rho_i nabla cdot v)"):
        return -stateA['densities'] * sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), operation = 'divergence', gradientMode='difference', neighborhood=neighborhood)

        i,j = neighborhood['indices']
        u_i = stateA['velocities'][i]
        u_j = stateB['velocities'][j]

        rho_j = stateB['densities'][j]

        q_ij = (u_j - u_i) * rho_j.view(-1,1)

        return -sphOperationStates(stateA, stateB, 
            q_ij, operation = 'divergence', gradientMode='difference', neighborhood=neighborhood)
