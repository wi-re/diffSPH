from diffSPH.v2.parameters import parseDefaultParameters, parseModuleParameters
from diffSPH.v2.sampling import sampleParticles
from diffSPH.v2.util import countUniqueEntries, printState
from diffSPH.v2.plotting import updatePlot, visualizeParticles, prepVisualizationState

from diffSPH.v2.modules.integration import integrate
# from diffSPH.v2.modules.neighborhood import fluidNeighborSearch
from diffSPH.v2.modules.shifting import solveShifting
from diffSPH.v2.modules.timestep import computeTimestep

from diffSPH.v2.simulationSchemes.deltaPlus import simulationStep    
from diffSPH.v2.modules.viscosity import computeViscosityParameter
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import os
import datetime
from diffSPH.v2.util import computeStatistics

def setViscosityParameter(config, targetRe):
    if config['diffusion']['velocityScheme'] == 'deltaSPH_inviscid':
        nu_sph = config['diffusion']['alpha'] * config['fluid']['cs'] * config['particle']['support']   / (2 * (config['domain']['dim'] + 2)) * 5/4
        Re = u_mag * (2 * L) / config['diffusion']['nu_sph']

        target_nu = u_mag * (2 * L) / targetRe
        alpha = target_nu / (config['fluid']['cs'] * config['particle']['support']  / (2 * (config['domain']['dim'] + 2)) * 5/4) #/ config['kernel']['kernelScale']
        config['diffusion']['alpha'] = alpha
        # print(alpha)
        if alpha < 0.01:
            print(rf'$\alpha = {alpha}$ is very low, consider increasing the value (should be > 0.01)')
    elif config['diffusion']['velocityScheme'] == 'deltaSPH_viscid':
        nu_sph = config['diffusion']['nu']
        Re = u_mag * (2 * L) / config['diffusion']['nu']
        target_nu = u_mag * (2 * L) / targetRe
        config['diffusion']['nu'] = target_nu

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--u_mag', type=float, default=2)
parser.add_argument('--k', type=float, default=1)
parser.add_argument('--TGV_override', type=bool, default=False)
parser.add_argument('--Re', type=float, default=2000)
parser.add_argument('--dt', type=float, default=0.001)
parser.add_argument('--L', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--baseFrequency', type=float, default=1)
parser.add_argument('--octaves', type=int, default=4)
parser.add_argument('--frameLimit', type=int, default=4096)
parser.add_argument("--verbose", help="increase output verbosity",)
args = parser.parse_args()


nx = args.nx
L = args.L
u_mag = args.u_mag
k = args.k * np.pi
TGV_override = args.TGV_override

from random import randint
seed = args.seed if args.seed != 0 else randint(0, 1000000)
if args.verbose:
    print('seed:', seed)

name = f'random_periodic_{nx}x{nx}_u{u_mag}_k{args.k}_TGV{TGV_override}_Re{args.Re}_dt{args.dt}_L{L}_seed{seed}_baseFrequency{args.baseFrequency}_octaves{args.octaves}_time{datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}'

config = {
    'domain': {
        'minExtent': -L,
        'maxExtent': L,
        'dim': 2,
        'periodic': True
    },
    'particle': {
        'nx': nx
    },
    'compute':{
        'device': 'cuda'
    },
    'fluid':{
        'cs' : u_mag * 10
    },
    'shifting':{
        'scheme': 'deltaSPH',
        'threshold': 0.05,
        'maxIterations': 1,
        'freeSurface': False
    },
    'timestep':{
        'dt':0.001,
        'active':False,
        'minDt':1e-6,
    },
    'sps':{
        'active': True
    },
    'noise':{
    #     'n': nx,
        'baseFrequency': args.baseFrequency,
    #     'dim': 2,
        'octaves': args.octaves,
    #     'persistence': 0.5,
    #     'lacunarity': 2,
        'seed': seed,
    #     'tileable': True,
    #     'kind': 'simplex',
    },
    'diffusion':{
        'velocityScheme': 'deltaSPH_inviscid'
    },
    'plot':{
    #     'mosaic': '''A''',
    #     'figSize': (6,5.5),
        'plots': {'A': {'val': 'index', 'cbar': True, 'cmap': 'twilight', 'scale': 'lin', 'size': 2, 'gridVis' : False, 'title': 'Fluid Index'}},
    #     'export': True,
        'updateInterval': 32*3,
        'namingScheme': name,
    #     'exportPath': 'output',
        # 'fps': 240,
        'exportFPS': 30,
    },
    'export':{
        'active': True,
    #     'path': 'export',
        'namingScheme': name,
    #     'interval': 1,
    },
    'boundary':{
        'active': False
    },
    'integration':{
        'scheme': 'RK4'
    }
    # 'simulation':{'timestamp': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}
}

config = parseDefaultParameters(config)
config = parseModuleParameters(config)
config['simulation']['correctArea'] = True


setViscosityParameter(config, args.Re)
config['diffusion']['nu_sph'] = computeViscosityParameter(None, config)# * config['kernel']['kernelScale']
config['diffusion']['Re'] = u_mag * (2 * L) / config['diffusion']['nu_sph']
if args.verbose:
    print(rf'Re = {config["diffusion"]["Re"]}, $\nu_{{sph}} = {config["diffusion"]["nu_sph"]}$')
    print(f'''Running random periodic simulation with arguments:
    particles: {nx} x {nx}
    u_mag: {u_mag}
    k: {k}
    TGV_override: {TGV_override}
    nu: {config['diffusion']['nu_sph']}
    Re: {config['diffusion']['Re']}
    dt: {args.dt}
    L: {L}
    seed: {seed}
    baseFrequency: {args.baseFrequency}
    octaves: {args.octaves}''')
# display(config)

# exit()

from diffSPH.v2.sampling import sampleNoisyParticles
from diffSPH.v2.modules.divergence import computeDivergence
from diffSPH.v2.plotting import plotSDF
from diffSPH.v2.sdf import getSDF, operatorDict


ngrid = 255
x = torch.linspace(-1, 1, ngrid, dtype = torch.float32)
y = torch.linspace(-1, 1, ngrid, dtype = torch.float32)
X, Y = torch.meshgrid(x, y, indexing = 'ij')
P = torch.stack([X,Y], dim=-1)
points = P.reshape(-1,2)

sphere_a = lambda points: getSDF('hexagon')['function'](points, 0.5)
sphere_b = lambda points: getSDF('box')['function'](points, torch.tensor([0.25,0.5]))
translated = operatorDict['translate'](sphere_b, torch.tensor([0.1,0.25]))
rotated = operatorDict['rotate'](translated, 0.5)
sdf = operatorDict['smooth_union'](rotated, sphere_a, 0.25)

# circle_a = operatorDict['translate'](lambda points: getSDF('hexagon')['function'](points, 0.35), torch.tensor([-0.25,0.]))
# circle_b = operatorDict['translate'](lambda points: getSDF('circle')['function'](points, 0.35), torch.tensor([ 0.25,0.]))
# sdf = operatorDict['smooth_union'](circle_a, circle_b, 0.05)
# sdf = operatorDict['twist'](sdf, 0.5)
# sdf = operatorDict['shell'](sdf, 0.125)
sdf = sphere_a
# plotSDF(sdf(torch.clone(points),), X, Y, 2, 2)

if args.verbose:
    print('Sampling particles')

noiseState, mask = sampleNoisyParticles(config['noise'], config, sdfs = [], randomizeParticles=False)

particleState = {
    'fluid': noiseState,

    'time': 0.0,
    'timestep': 0,
    'dt': config['timestep']['dt'],
}

priorState = None


if TGV_override:
    if args.verbose:
        print('Overriding with TGV')
    k = args.k * np.pi
    particleState['fluid']['velocities'][:,0] =  u_mag * torch.cos(k * particleState['fluid']['positions'][:,0]) * torch.sin(k * particleState['fluidPositions'][:,1])
    particleState['fluid']['velocities'][:,1] = -u_mag * torch.sin(k * particleState['fluid']['positions'][:,0]) * torch.cos(k * particleState['fluidPositions'][:,1])

# omega = 4

# k = np.pi
if args.verbose:
    print('Setting initial velocities')

u_max = torch.linalg.norm(particleState['fluid']['velocities'], dim = 1).max()
particleState['fluid']['velocities'] = particleState['fluid']['velocities'] / u_max * u_mag

Ek0 = 0.5 * particleState['fluid']['areas'] * particleState['fluid']['densities'] * torch.linalg.norm(particleState['fluid']['velocities'], dim = -1)**2

targetEK = 3000
ratio = np.sqrt(targetEK / Ek0.detach().sum().cpu().item())
# print(f'ratio: {ratio}')
# particleState['fluidVelocities'] = particleState['fluidVelocities'] * ratio

initialVelocities = particleState['fluid']['velocities'].clone()


particleState['fluid']['Eks'] =  (0.5 * particleState['fluid']['areas'] * particleState['fluid']['densities'] * torch.linalg.norm(initialVelocities, dim = -1)**2)
particleState['fluid']['E_k'] = particleState['fluid']['Eks'].sum().cpu().detach().item()

perennialState = copy.deepcopy(particleState)


from diffSPH.v2.simulationSchemes.dfsph import dfsphSimulationStep
from diffSPH.v2.runner import runSimulation, setupSimulation

if args.verbose:
    print('Setting up simulation')
perennialState, fig, axis, plotStates, priorState, pbar, stats = setupSimulation(particleState, config, stepLimit = args.frameLimit, timeLimit = -1.0)

if args.verbose:
    print('Running simulation')
stat, pstate = runSimulation(fig, axis, dfsphSimulationStep, plotStates, priorState, pbar, stats, perennialState, particleState, config, stepLimit = args.frameLimit, timeLimit = -1.0)

if args.verbose:
    print('Simulation done')