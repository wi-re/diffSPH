import torch
import numpy as np

@torch.jit.script
def countUniqueEntries(indices, positions):
    """
    Count the number of unique entries in the indices tensor and return the unique indices and their counts.

    Args:
        indices (torch.Tensor): Tensor containing the indices.
        positions (torch.Tensor): Tensor containing the positions.

    Returns:
        tuple: A tuple containing the unique indices and their counts.
    """
    ii, nit = torch.unique(indices, return_counts=True)
    ni = torch.zeros(positions.shape[0], dtype=nit.dtype, device=positions.device)
    ni[ii] = nit
    return ii, ni



@torch.jit.script
def volumeToSupport(volume : float, targetNeighbors : float, dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    torch.Tensor: The support radius.
    """
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        return torch.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)
    

def printState(particleState):
    for k in particleState.keys():
        if isinstance(particleState[k], torch.Tensor):
            print(f'state[{k:24s}]: min: {particleState[k].min():+.3e}, max: {particleState[k].max():+.3e}, median: {particleState[k].median():+.3e} [{particleState[k].shape}\t@ {particleState[k].device}\tx {particleState[k].dtype}\t]')
        elif isinstance(particleState[k], dict):
            print(f'state[{k:24s}]: dict')
            for kk in particleState[k].keys():
                if isinstance(particleState[k][kk], torch.Tensor):
                    print(f'state[{k:24s}][{kk:18s}]: min: {particleState[k][kk].min():+.3e}, max: {particleState[k][kk].max():+.3e}, median: {particleState[k][kk].median():+.3e} [{particleState[k][kk].shape}\t@ {particleState[k][kk].device}\tx {particleState[k][kk].dtype}\t]')
                else:
                    print(f'state[{k:24s}][{kk:18s}]: {particleState[k][kk]}\t[{type(particleState[k][kk])}]')        
        else:
            if particleState[k] is None:
                print(f'state[{k:24s}]: None')
            else:
                print(f'state[{k:24s}]: {particleState[k]:10}\t[{type(particleState[k])}]')


def computeStatistics(perennialState, particleState, config, print = False):
    E_k0 = particleState['E_k']
    E_k = (0.5 * torch.sum(perennialState['fluidAreas'] * perennialState['fluidDensities'] * torch.linalg.norm(perennialState['fluidVelocities'], dim = 1)**2)).sum().detach().cpu().numpy()
    if print:
        print(f'E_k0 = {E_k0:.4g}, E_k = {E_k:.4g}, rel. diff = {(E_k - E_k0)/E_k0:.2%}')

    c = perennialState['fluidDensities'] / config['fluid']['rho0'] - 1
    maxCompression = c.max().cpu().detach().item()
    minCompression = c.min().cpu().detach().item()

    if print:
        print(f'Max compression: {maxCompression*100:.4g}%, min compression: {minCompression*100:.4g}%')

    averageDensity = perennialState['fluidDensities'].mean().cpu().detach().item() / config['fluid']['rho0']
    if print:
        print(f'Average density: {averageDensity:.4g}')
    averageCompression = averageDensity - 1
    if print:
        print(f'Average compression: {averageCompression*100:.4g}%')
    if 'fliudShiftAmount' in perennialState:
        dx = perennialState['fluidShiftAmount']
        shiftAmount = torch.linalg.norm(dx, dim = -1)#.max().cpu().detach().item()
        maxShift = shiftAmount.max().cpu().detach().item() /  config['particle']['dx']
        if print:
            print(f'Max shift: {maxShift*100:.4g}%')
    else:
        maxShift = torch.tensor(0)

    dt_v = 0.125 * config['particle']['support']**2 / config['diffusion']['nu_sph'] / config['kernel']['kernelScale']**2
    # acoustic timestep condition

    dt_c = config['timestep']['CFL'] * config['particle']['support'] / config['fluid']['cs'] / config['kernel']['kernelScale']    
    # print(dt_v, dt_c)
    if 'fluid_dudt' in perennialState:
        dudt = perennialState['fluid_dudt']
        max_accel = torch.max(torch.linalg.norm(dudt[~torch.isnan(dudt)], dim = -1))
    else:
        max_accel = 0

    dt_a = 0.25 * torch.sqrt(config['particle']['support'] / (max_accel + 1e-7)) / config['kernel']['kernelScale']

    if print:
        print(f'dt_v = {dt_v:.4g}, dt_c = {dt_c:.4g}, dt_a = {dt_a:.4g}')

    CFLNumber = config['timestep']['dt'] / (config['particle']['support'] / config['fluid']['cs'] / config['kernel']['kernelScale'])
    if print:
        print(f'CFL number: {CFLNumber:.4g}')

    minNeighborCount = perennialState['fluidNumNeighbors'].min().cpu().detach().item()
    maxNeighborCount = perennialState['fluidNumNeighbors'].max().cpu().detach().item()
    if print:
        print(f'Min neighbors: {minNeighborCount}, max neighbors: {maxNeighborCount}')

    medianNeighborCount = torch.median(perennialState['fluidNumNeighbors']).cpu().detach().item()
    if print:
        print(f'Median neighbors: {medianNeighborCount}')

    frameStatistics = {
        'E_k': E_k.item(),
        'maxCompression': maxCompression,
        'minCompression': minCompression,
        'averageDensity': averageDensity,
        'averageCompression': averageCompression,
        'maxShift': maxShift.cpu().detach().item(),
        'dt_v': dt_v.cpu().detach().item(),
        'dt_c': dt_c.cpu().detach().item(),
        'dt_a': dt_a.cpu().detach().item(),
        'CFLNumber': CFLNumber.cpu().detach().item(),
        'minNeighborCount': minNeighborCount,
        'maxNeighborCount': maxNeighborCount,
        'medianNeighborCount': medianNeighborCount,
        'time': perennialState['time'] if not isinstance(perennialState['time'], torch.Tensor) else perennialState['time'].cpu().item(),
        'timestep': perennialState['timestep'],
        'dt': perennialState['dt'] if not isinstance(perennialState['dt'], torch.Tensor) else perennialState['dt'].cpu().item()
    }
    return frameStatistics