import torch
from diffSPH.v2.sphOps import sphOperationFluidState, sphOperation
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from torch.profiler import record_function

from diffSPH.v2.math import mod

def computeGravity(fluidState, config):
    if not config['gravity']['active']:
        return torch.zeros_like(fluidState['fluidVelocities'])
    with record_function("[SPH] - External Gravity Field (g)"):
        if config['gravity']['gravityMode'] == 'potentialField':
            x = fluidState['fluidPositions']
            minD = config['domain']['minExtent']
            maxD = config['domain']['maxExtent']
            periodicity = config['domain']['periodicity']

            center = torch.tensor(config['gravity']['potentialOrigin'], dtype = x.dtype, device = x.device)[:x.shape[-1]]

            xij = torch.stack([x[:,i] - center[i] if not periodic_i else mod(x[:,i] - center[i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
            rij = torch.linalg.norm(xij, dim = -1)
            xij[rij > 1e-7] = xij[rij > 1e-7] / rij[rij > 1e-7, None]

            magnitude = config['gravity']['magnitude']
            return - magnitude**2 * xij * (rij)[:,None] #/ fluidState['fluidDensities'][:,None]
        else:
            v = fluidState['fluidVelocities']
            direction = torch.tensor(config['gravity']['direction'], dtype = fluidState['fluidPositions'].dtype, device = fluidState['fluidPositions'].device)
            return (direction[:v.shape[1]] * config['gravity']['magnitude']).repeat(v.shape[0], 1)


    
from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('gravity', 'active', bool, False, required = False,export = True, hint = 'Enables/disables gravity'),
        # Parameter('gravity', 'g', float, 9.81, required = False,export = False, hint = 'The acceleration due to gravity'), 
        Parameter('gravity', 'direction', list, [0, -1, 0], required = False,export = True, hint = 'The direction of the gravity vector'),
        Parameter('gravity', 'magnitude', float, 9.81, required = False,export = True, hint = 'The magnitude of the gravity vector'),
        Parameter('gravity', 'gravityMode', str, 'constant', required = False,export = True, hint = 'The mode of the gravity vector. Options are: constant, potentialField'),
        Parameter('gravity', 'potentialOrigin', list, [0,0,0], required = False,export = True, hint = 'The origin of the potential field'),
    ]