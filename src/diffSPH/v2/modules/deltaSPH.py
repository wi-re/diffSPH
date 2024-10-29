import torch
from torch.profiler import record_function
from diffSPH.v2.math import scatter_sum

def computeDeltaU(particleState, config):
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
        densityTerm = particleState['masses'][j] / (particleState['densities'][j])
        phi_ij = 1

        scalarTerm = term * densityTerm * phi_ij
        shiftAmount = scatter_sum(scalarTerm.view(-1,1) * gradK, i, dim = 0, dim_size = particleState['positions'].shape[0])

        CFL = config['shifting']['CFL']
        if config['shifting']['computeMach'] == False:
            Ma = 0.1
        else:
            Ma = torch.amax(torch.linalg.norm(particleState['velocities'], dim = -1)) #/ config['fluid']['cs']
        shiftScaling = - Ma * (particleState['supports'] / config['kernel']['kernelScale'] * 2)#**2
        # print(f'Shift: {shiftAmount.abs().max()}, Scaling: {shiftScaling.shape}')
        # print(particleState['fluidSupports'])
        shiftScale = torch.linalg.norm(shiftAmount, dim = -1) * shiftScaling
        clampedShiftScale = torch.min(shiftScale, Ma / 2)

        return  shiftAmount * ( clampedShiftScale / (shiftScale + 1e-7)).view(-1,1)
