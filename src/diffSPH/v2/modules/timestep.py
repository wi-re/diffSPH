import torch
from diffSPH.v2.sphOps import sphOperationFluidState, sphOperation
from diffSPH.v2.modules.normalizationMatrices import computeNormalizationMatrices
from torch.profiler import record_function


# See Sun et al: The delta SPH-model: Simple procedures for a further improvement of the SPH scheme
def computeTimestep(perennialState, config):
    if not config['timestep']['active']:
        return config['timestep']['dt']
    with record_function("[SPH] - Adaptive Timestep Update"):
        # timestep condition due to viscosity
        nu = config['diffusion']['alpha'] * config['fluid']['cs'] * config['particle']['support'] / (2 * (config['domain']['dim'] +2))
        nu = config['diffusion']['nu'] if config['diffusion']['velocityScheme'] == 'deltaSPH_viscid' else nu
        dt_v = 0.125 * config['particle']['support']**2 / nu / config['kernel']['kernelScale']**2
        # acoustic timestep condition

        dt_c = config['timestep']['CFL'] * config['particle']['support'] / config['fluid']['cs'] / config['kernel']['kernelScale']    
        # print(dt_v, dt_c)

        # acceleration timestep condition
        if 'fluid_dudt' in perennialState: 
            dudt = perennialState['fluid_dudt']
            max_accel = torch.max(torch.linalg.norm(dudt[~torch.isnan(dudt)], dim = -1))
            dt_a = 0.25 * torch.sqrt(config['particle']['support'] / (max_accel + 1e-7)) / config['kernel']['kernelScale']
        else:
            dt_a = config['timestep']['maxDt']

        dt = config['timestep']['dt']
        new_dt = dt
        if config['timestep']['viscosityConstraint']:
            new_dt = dt_v
        if config['timestep']['accelerationConstraint']:
            new_dt = torch.min(new_dt, dt_a)
        if config['timestep']['acousticConstraint']:
            new_dt = torch.min(new_dt, dt_c)
        new_dt = torch.min(new_dt, torch.tensor(config['timestep']['maxDt'], dtype = new_dt.dtype, device = new_dt.device))
        new_dt = torch.max(new_dt, torch.tensor(config['timestep']['minDt'], dtype = new_dt.dtype, device = new_dt.device))
        return new_dt



    
from diffSPH.parameter import Parameter
def getParameters():
    return [
    Parameter('timestep', 'dt', float, 0.001, required = False, export = True),
    Parameter('timestep', 'active', bool, True, required = False, export = True),
    Parameter('timestep', 'CFL', float, 1.5, required = False, export = True),
    Parameter('timestep', 'maxDt', float, 0.1, required = False, export = True),
    Parameter('timestep', 'minDt', float, 0.001, required = False, export = True),
    Parameter('timestep', 'viscosityConstraint', bool, True, required = False, export = True),
    Parameter('timestep', 'accelerationConstraint', bool, True, required = False, export = True),
    Parameter('timestep', 'acousticConstraint', bool, True, required = False, export = True),

]