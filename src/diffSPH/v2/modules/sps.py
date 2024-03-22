from diffSPH.v2.math import scatter_sum
import torch
from math import sqrt
from diffSPH.v2.sphOps import sphOperationStates

def computeSPSTurbulence(stateA, stateB, neighborhood, config):
    if config['SPS']['active'] == False:
        return torch.zeros_like(stateA['velocities'])
    velGrad = -sphOperationStates(stateA, stateB, (stateA['velocities'], stateB['velocities']), 'gradient', 'difference')
    # - from dual sphysics

    u_xx = velGrad[:,0,0]
    u_xy = velGrad[:,0,1] + velGrad[:,1,0]
    u_yy = velGrad[:,1,1]
    # modification from dual sphysics

    C_smagorinsky = config['SPS']['Smagorinsky']
    C_blinn = config['SPS']['Blinn'] # dual sphysics uses 0.0066
    
    dx = config['particle']['dx']
    dx_sps = dx / sqrt(config['domain']['dim'])

    Sps_smag = C_smagorinsky**2 * dx_sps**2
    Sps_blinn = 2/3 * C_blinn * dx_sps**2

    pow1 = u_xx**2 + u_yy**2
    prr = pow1 + pow1 + u_xy**2

    visc_sps = Sps_smag * torch.sqrt(prr)
    div_u = velGrad.diagonal(dim1 = -2, dim2 = -1).sum(dim = -1)
    sps_k = 2/3 * visc_sps * div_u
    sps_blinn = Sps_blinn * prr
    sumsps = -(sps_k + sps_blinn)

    onerho = 1 / stateA['densities']
    tau_xx = onerho * (2 * visc_sps * u_xx + sumsps)
    tau_xy = onerho * (visc_sps * u_xy)
    tau_yy = onerho * (2 * visc_sps * u_yy + sumsps)

    m = stateA['masses']
    (i,j) = neighborhood['indices']
    gradWij = neighborhood['gradients']

    dudt_x = m[j] * ((tau_xx[i] + tau_xx[j]) * gradWij[:,0] + (tau_xy[i] + tau_xy[j]) * gradWij[:,1])
    dudt_y = m[j] * ((tau_xy[i] + tau_xy[j]) * gradWij[:,0] + (tau_yy[i] + tau_yy[j]) * gradWij[:,1])
    dudt = torch.stack([dudt_x, dudt_y], dim = 1)
    dudt = scatter_sum(dudt, i, dim = 0, dim_size = stateA['positions'].shape[0])

    return dudt 


from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('SPS', 'active', bool, False, required = False,export = False, hint = 'Activate the SPS turbulence model'),
        Parameter('SPS', 'Smagorinsky', float, 0.12, required = False,export = False, hint = 'Smagorinsky constant'),
        Parameter('SPS', 'Blinn', float, 0.0066, required = False,export = False, hint = 'Blinn constant'),
    ]