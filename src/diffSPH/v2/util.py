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
            print(f'state[{k:24s}]: {particleState[k]:10}\t[{type(particleState[k])}]')