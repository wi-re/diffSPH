# %matplotlib widget
import torch
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

from diffSPH.v2.parameters import parseDefaultParameters, parseModuleParameters
from diffSPH.v2.sampling import sampleParticles
from diffSPH.v2.util import countUniqueEntries, printState
from diffSPH.v2.plotting import updatePlot, visualizeParticles, prepVisualizationState

from diffSPH.v2.modules.integration import integrate
from diffSPH.v2.modules.neighborhood import neighborSearch
from diffSPH.v2.modules.shifting import solveShifting
from diffSPH.v2.modules.timestep import computeTimestep

from diffSPH.v2.simulationSchemes.deltaPlus import simulationStep    
from diffSPH.v2.modules.viscosity import computeViscosityParameter, setViscosityParameters
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import os
import datetime
from diffSPH.v2.util import computeStatistics
import tomli
from torch.profiler import profile, record_function, ProfilerActivity
from diffSPH.v2.modules.inletOutlet import processInlet, processOutlet, processForcing
from diffSPH.v2.plotting import plotRegions
from diffSPH.v2.sdf import getSDF, sdfFunctions, operatorDict
from diffSPH.v2.sampling import find_contour
from diffSPH.v2.plotUtil import plotRegionswSDF
from diffSPH.v2.sampling import sampleParticles, processBoundarySDFs, sampleNoisyParticles
from diffSPH.v2.finiteDifference import centralDifferenceStencil, continuousGradient
from diffSPH.v2.plotUtil import plotInitialParticleSampling
from diffSPH.v2.modules.neighborhood import searchNeighbors
from diffSPH.v2.plotting import visualizeParticleQuantity
from diffSPH.v2.sampling import generateRamp, sampleVelocityField
from diffSPH.v2.modules.neighborhood import searchNeighbors
from diffSPH.v2.sampling import generateRamp, getNoiseSampler
from diffSPH.v2.plotUtil import plotPotentialField
from diffSPH.v2.plotUtil import plotBoundaryParticleswGhosts
from diffSPH.v2.modules.mDBC import mDBCDensity
from diffSPH.v2.modules.boundaryViscosity import computeBoundaryVelocities
from diffSPH.v2.plotUtil import plotBoundaryVelocityFields
from diffSPH.v2.simulationSchemes.deltaPlus import simulationStep, callModule
from diffSPH.v2.modules.density import computeDensity
from diffSPH.v2.plotting import exportPlot, postProcessPlot
from diffSPH.v2.simulationSchemes.deltaPlus import checkNaN
from diffSPH.v2.modules.mDBC import buildBoundaryGhostParticles
from diffSPH.v2.modules.boundaryViscosity import computeBoundaryVelocities
from diffSPH.v2.plotUtil import plotBoundaryParticleswGhosts

configurationFile = './configurations/05_ldc.toml'

with open(configurationFile, 'r') as f:
    config = f.read()

config = tomli.loads(config)
config['compute'] = {'device': 'cuda', 'checkNaN': True}



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--u_mag', type=float, default=2)
parser.add_argument('--k', type=float, default=2)
parser.add_argument('--TGV_override', type=bool, default=False)
parser.add_argument('--Re', type=float, default=2000)
parser.add_argument('--dt', type=float, default=0.001)
parser.add_argument('--L', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--baseFrequency', type=float, default=1)
parser.add_argument('--octaves', type=int, default=4)
parser.add_argument('--frameLimit', type=int, default=4096)
parser.add_argument("--verbose", help="increase output verbosity",)
parser.add_argument('--boundary', type=str, default='free-slip')
parser.add_argument('--centerObstacle', type=bool, default=False)
args = parser.parse_args()

if args.boundary == 'none':
    config['domain']['periodic'] = True
    config['boundary']['boundary_width'] = 0
    config['boundary']['active'] = False

config['particle']['nx'] = args.nx
config['fluid'] = {'u_mag' : args.u_mag}
config['diffusion']['targetRe'] = args.Re



config = parseDefaultParameters(config)
config = parseModuleParameters(config)
# config['diffusion']['alpha'] = 0.01



ngrid = 255
x = torch.linspace(config['domain']['minExtent'][0], config['domain']['maxExtent'][0], ngrid, dtype = torch.float32)
y = torch.linspace(config['domain']['minExtent'][1], config['domain']['maxExtent'][1], ngrid, dtype = torch.float32)
X, Y = torch.meshgrid(x, y, indexing = 'ij')
P = torch.stack([X,Y], dim=-1)
points = P.reshape(-1,2)


sphere_b = lambda points: getSDF('box')['function'](points, torch.tensor([1/8,1/2]).to(points.device))
inletSDF = operatorDict['translate'](sphere_b, torch.tensor([-2 + 1/8,0.]))
# fx = sdf(torch.clone(points)).to('cpu').reshape(ngrid,ngrid)


sphere_b = lambda points: getSDF('box')['function'](points, torch.tensor([1/8,1]).to(points.device))
outletSDF = operatorDict['translate'](sphere_b, torch.tensor([2 - 1/8,0.]))
# fx = sdf(torch.clone(points)).to('cpu').reshape(ngrid,ngrid)

sphere_b = lambda points: getSDF('box')['function'](points, torch.tensor([1/4,1.5]).to(points.device))
outletRegion = operatorDict['translate'](sphere_b, torch.tensor([2 - 1/4,0.]))
# fx = sdf(torch.clone(points)).to('cpu').reshape(ngrid,ngrid)

inner = lambda points: getSDF('box')['function'](points, torch.tensor([0.5,0.5]).to(points.device))
outer = lambda points: getSDF('box')['function'](points, torch.tensor([0.5 + config['particle']['dx'] * config['boundary']['boundary_width'],0.5 + config['particle']['dx'] * config['boundary']['boundary_width']]).to(points.device))
outer = operatorDict['invert'](outer)
sdf = operatorDict['union'](inner, outer)
boundary_sdf = operatorDict['invert'](inner)

l = config['particle']['dx'] * 20

inner = lambda points: getSDF('box')['function'](points, torch.tensor([0.15,0.15]).to(points.device))
boundary_sdf2 = inner# operatorDict['invert'](inner)

circle_sdf = lambda points: getSDF('circle')['function'](points, torch.tensor(l).to(points.device))
translated_sdf = operatorDict['translate'](circle_sdf, torch.tensor([0.,0.]))
boundary_sdf2 = lambda points: translated_sdf(points.cpu()).to(points.device)

fluid_sdf = lambda points: getSDF('box')['function'](points, torch.tensor([1/2, 1/2]).to(points.device))
# fluid_sdf = operatorDict['translate'](sphere_b, torch.tensor([-2 + 1/2, - 1/2]))

if args.boundary != 'none':
    if args.centerObstacle:
        regions = [
            {
                'sdf': boundary_sdf,
                'type': 'boundary',
                'velocity': torch.tensor([0.,0.]),
                'kind': 'zero',
                'particles': sampleParticles(config, sdfs = [boundary_sdf])[0],
                'contour': find_contour(boundary_sdf(points).reshape(ngrid, ngrid).cpu(), config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), ngrid, 0)
            },
            {
                'sdf': boundary_sdf2,
                'type': 'boundary',
                'kind': 'free-slip',
                'velocity': torch.tensor([0.,0.]),
                'particles': sampleParticles(config, sdfs = [boundary_sdf2])[0],
                'contour': find_contour(boundary_sdf2(points).reshape(ngrid, ngrid).cpu(), config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), ngrid, 0)
            },
            {
                'sdf': fluid_sdf,
                'type': 'fluid',
                'velocity': torch.tensor([0.,0.]),
                'particles': sampleParticles(config, sdfs = [fluid_sdf])[0],
                'contour': find_contour(fluid_sdf(points).reshape(ngrid, ngrid).cpu(), config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), ngrid, 0)
            }
            ]
    else:
        regions = [
            {
                'sdf': boundary_sdf,
                'type': 'boundary',
                'velocity': torch.tensor([0.,0.]),
                'kind': 'zero',
                'particles': sampleParticles(config, sdfs = [boundary_sdf])[0],
                'contour': find_contour(boundary_sdf(points).reshape(ngrid, ngrid).cpu(), config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), ngrid, 0)
            },
            {
                'sdf': fluid_sdf,
                'type': 'fluid',
                'velocity': torch.tensor([0.,0.]),
                'particles': sampleParticles(config, sdfs = [fluid_sdf])[0],
                'contour': find_contour(fluid_sdf(points).reshape(ngrid, ngrid).cpu(), config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), ngrid, 0)
            }
            ]

else:
    regions = [
        {
            'sdf': fluid_sdf,
            'type': 'fluid',
            'velocity': torch.tensor([0.,0.]),
            'particles': sampleParticles(config, sdfs = [fluid_sdf])[0],
            'contour': find_contour(fluid_sdf(points).reshape(ngrid, ngrid).cpu(), config['domain']['minExtent'].cpu(), config['domain']['maxExtent'].cpu(), ngrid, 0)
        }
    ]

config['regions'] = regions

for region in config['regions']:
    if region['type'] == 'boundary':
        region['kind'] = args.boundary
        # region['kind'] = 'zero'
        # region['kind'] = 'linear'
        
# plotRegionswSDF(config)

particleState, mask = sampleParticles(config, sdfs = [region['sdf'] for region in regions if region['type'] == 'fluid'], filter = True)
if args.boundary != 'none':
    boundaryState = processBoundarySDFs(particleState, config, [region['sdf'] for region in regions if region['type'] == 'boundary'], 'regular')
else:
    boundaryState = None

perennialState = {
    'fluid': copy.deepcopy(particleState),
    'boundary': boundaryState,
    'time': 0.0,
    'timestep': 0,
    'dt': config['timestep']['dt'],
    'uidCounter': particleState['numParticles']
}

u = 1

# perennialState['fluid']['positions'] += torch.normal(mean = 0, std = config['particle']['dx'] * 0.01, size = [perennialState['fluid']['numParticles'], 2], device = perennialState['fluid']['positions'].device)
# if args.boundary != 'none': 
    # perennialState['boundary']['positions'] += torch.normal(mean = 0, std = config['particle']['dx'] * 0.01, size = [perennialState['boundary']['numParticles'], 2], device = perennialState['boundary']['positions'].device)

searchNeighbors(perennialState, config)
# plotInitialParticleSampling(perennialState, config)
# plotBoundaryParticleswGhosts(perennialState, config)
if args.boundary == 'none': 
    ramp = torch.ones_like(perennialState['fluid']['positions'][:,0])
else:
    ramp = generateRamp(perennialState['fluid'], regions, config)

config['noise']['baseFrequency'] = args.baseFrequency
config['noise']['octaves'] = args.octaves
config['noise']['seed'] = args.seed


timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


# display(config['noise'])

x = perennialState['fluid']['positions']
# random sample
noiseSampler = getNoiseSampler(config)
potential = noiseSampler(perennialState['fluid']['positions'])


# return
# offset TGV
config['plot']['namingScheme'] = f'output_{args.boundary}{"_wObstacle" if args.centerObstacle else ""}_{args.nx}x{args.u_mag}x{args.baseFrequency}x{args.octaves}x{args.seed}_{timestamp}'

if args.TGV_override:
    potential = torch.sin(np.pi * args.k * perennialState['fluid']['positions'][:,0]) * torch.sin(np.pi * args.k * perennialState['fluid']['positions'][:,1]) / 6

    config['plot']['namingScheme'] = f'output_{args.boundary}{"_wObstacle" if args.centerObstacle else ""}_TGV_{args.nx}x{args.u_mag}x{args.k}_{timestamp}'


# TGV Potential field
# potential = torch.sin(np.pi * 2 * perennialState['fluid']['positions'][:,0]) * torch.sin(np.pi * 2 * perennialState['fluid']['positions'][:,1]) / 6
ramped = potential * ramp

filteredState = copy.deepcopy(perennialState)
filteredState['fluid']['potential'] = ramped

velocity, divergence = sampleVelocityField(filteredState['fluid'], filteredState['fluid']['neighborhood'])
u_max = torch.linalg.norm(velocity, dim = 1).max() + 1e-6
u_factor = config['fluid']['u_mag'] / u_max
velocity = velocity * u_factor
perennialState['fluid']['velocities'] = velocity

if args.TGV_override:
    k = args.k * np.pi
    perennialState['fluid']['velocities'][:,0] =  config['fluid']['u_mag'] * torch.cos(k * perennialState['fluid']['positions'][:,0]) * torch.sin(k * perennialState['fluid']['positions'][:,1])
    perennialState['fluid']['velocities'][:,1] = -config['fluid']['u_mag'] * torch.sin(k * perennialState['fluid']['positions'][:,0]) * torch.cos(k * perennialState['fluid']['positions'][:,1])



# plotPotentialField(perennialState, config, potential, ramped, s = 1)
# plotBoundaryVelocityFields(perennialState, config)
initialState = copy.deepcopy(perennialState)

if args.boundary == 'none':
    del initialState['boundary']

perennialState = copy.deepcopy(initialState)
perennialState['fluid']['pressureAccel'] = torch.zeros_like(perennialState['fluid']['velocities'])
perennialState, priorState, *updates = integrate(simulationStep, perennialState, config, previousStep= None)

fig, axis = plt.subplots(1,4, figsize = (14,4), squeeze = False)

visualizationState = prepVisualizationState(perennialState, config, grid = True)
plotRegions(regions, axis[0,0], plotParticles=False)
plotRegions(regions, axis[0,1], plotParticles=False)
plotRegions(regions, axis[0,2], plotParticles=False)
plotRegions(regions, axis[0,3], plotParticles=False)

s = 0.5

uxPlot = visualizeParticleQuantity(fig, axis[0,1], config, visualizationState, quantity = 'velocities', mapping = '.x', s = s,  scaling = 'sym', gridVisualization=True, cmap = 'RdBu_r', title = 'x-Velocity', which = 'fluid', plotBoth = False, midPoint = 0)
uyPlot = visualizeParticleQuantity(fig, axis[0,2], config, visualizationState, quantity = 'velocities', mapping = '.y', s = s, scaling = 'sym', gridVisualization=True, cmap = 'RdBu_r', streamLines = False, operation = None, title = 'y-Velocity', plotBoth = False, which = 'fluid', midPoint = 0)

indexPlot = visualizeParticleQuantity(fig, axis[0,0], config, visualizationState, quantity = 'index', mapping = None, s = s, scaling = 'lin', gridVisualization=False, cmap = 'twilight_r', streamLines = False, operation = None, title = 'Indices', plotBoth = False, which = 'fluid', midPoint = 1000)

# pressurePlot = visualizeParticleQuantity(fig, axis[0,1], config, visualizationState, quantity = 'densities', mapping = 'L2', s = s, scaling = 'lin', gridVisualization=True, cmap = 'magma', streamLines = False, operation = None, title = '$\\rho$', plotBoth = False, which = 'fluid', midPoint = 1000)

uPlot = visualizeParticleQuantity(fig, axis[0,3], config, visualizationState, quantity = 'velocities',mapping = 'L2', s = s, scaling = 'lin', gridVisualization=True, cmap = 'viridis', streamLines = True, operation = None, title = 'Velocities', plotBoth = True, which = 'fluid', midPoint = 1000)


Ek_0 = 0.5 * perennialState['fluid']['masses'] * torch.sum(velocity**2, dim = 1)
Ek = 0.5 * perennialState['fluid']['masses'] * torch.sum(perennialState['fluid']['velocities']**2, dim = 1)

nu = config['diffusion']['alpha'] * config['fluid']['cs'] * config['particle']['support'] / (2 * (config['domain']['dim'] +2))
nu = config['diffusion']['nu'] if config['diffusion']['velocityScheme'] == 'deltaSPH_viscid' else nu
dt_v = 0.125 * config['particle']['support']**2 / nu / config['kernel']['kernelScale']**2
# acoustic timestep condition

dt_c = config['timestep']['CFL'] * config['particle']['support'] / config['fluid']['cs'] / config['kernel']['kernelScale']    
# print(dt_v, dt_c)

# acceleration timestep condition
if 'dudt' in perennialState['fluid']: 
    dudt = perennialState['fluid']['dudt']
    max_accel = torch.max(torch.linalg.norm(dudt[~torch.isnan(dudt)], dim = -1))
    dt_a = 0.25 * torch.sqrt(config['particle']['support'] / (max_accel + 1e-7)) / config['kernel']['kernelScale']
else:
    dt_a = torch.tensor(config['timestep']['maxDt'], dtype = dt_v.dtype, device = dt_v.device)

if args.boundary == 'none':
    fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid) E_k: {Ek.sum()}, Ratio: {Ek.sum() / Ek_0.sum() * 100 : .2f}\n$\\Delta t$ = {perennialState["dt"]} $\\Delta t_\\nu$ = {dt_v:.3g}, $\\Delta t_\\text{{CFL}}$ = {dt_c:.3g}, $\\Delta t_a$ = {dt_a:.3g} $\\bar\\rho$ = {perennialState["fluid"]["densities"].mean():.2f}')

else:
    fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"] + perennialState["boundary"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid + {perennialState["boundary"]["numParticles"]} boundary) E_k: {Ek.sum()}, Ratio: {Ek.sum() / Ek_0.sum() * 100 : .2f}\n$\\Delta t$ = {perennialState["dt"]} $\\Delta t_\\nu$ = {dt_v:.3g}, $\\Delta t_\\text{{CFL}}$ = {dt_c:.3g}, $\\Delta t_a$ = {dt_a:.3g} $\\bar\\rho$ = {perennialState["fluid"]["densities"].mean():.2f}')

fig.tight_layout()

perennialState = copy.deepcopy(initialState)
perennialState['fluid']['pressureAccel'] = torch.zeros_like(perennialState['fluid']['velocities'])

# config["simulation"]["timestamp"] = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

from diffSPH.v2.io import createOutputFile, writeFrame
from diffSPH.v2.io import createOutputFile
def initializeOutput(config, initialState):

    f, simulationDataGroup = createOutputFile(config)
    f.attrs.version = 'diffSPHv2'

    initialGroup = f.create_group('initial')
    initialFluid = initialGroup.create_group('fluid')
    initialFluid.create_dataset('positions', data = initialState['fluid']['positions'].detach().cpu().numpy())
    initialFluid.create_dataset('velocities', data = initialState['fluid']['velocities'].detach().cpu().numpy())
    initialFluid.create_dataset('areas', data = initialState['fluid']['areas'].detach().cpu().numpy())
    initialFluid.create_dataset('masses', data = initialState['fluid']['masses'].detach().cpu().numpy())
    initialFluid.create_dataset('densities', data = initialState['fluid']['densities'].detach().cpu().numpy() / config['fluid']['rho0'])
    initialFluid.create_dataset('UID', data = initialState['fluid']['index'].detach().cpu().numpy())
    initialFluid.create_dataset('potential', data = ramped.detach().cpu().numpy())

    if 'boundary' in initialState and initialState['boundary'] is not None:
        initialBoundary = initialGroup.create_group('boundary')
        initialBoundary.create_dataset('positions', data = initialState['boundary']['positions'].detach().cpu().numpy())
        initialBoundary.create_dataset('velocities', data = initialState['boundary']['velocities'].detach().cpu().numpy())
        initialBoundary.create_dataset('areas', data = initialState['boundary']['areas'].detach().cpu().numpy())
        initialBoundary.create_dataset('masses', data = initialState['boundary']['masses'].detach().cpu().numpy())
        initialBoundary.create_dataset('densities', data = initialState['boundary']['densities'].detach().cpu().numpy() / config['fluid']['rho0'])
        initialBoundary.create_dataset('distances', data = initialState['boundary']['distances'].detach().cpu().numpy())
        initialBoundary.create_dataset('normals', data = initialState['boundary']['normals'].detach().cpu().numpy())
        initialBoundary.create_dataset('bodyIDs', data = initialState['boundary']['bodyIDs'].detach().cpu().numpy())
        initialBoundary.create_dataset('UID', data = initialState['boundary']['index'].detach().cpu().numpy())

    return f, simulationDataGroup
def exportFrame(pernenialState, particleState, config, simulationDataGroup, dynamicBoundary = False):
    frameStatistics = computeStatistics(perennialState, particleState, config)
    frameGroup = simulationDataGroup.create_group(f'{perennialState["timestep"]:05d}')

    for key in frameStatistics:
        frameGroup.attrs[key] = frameStatistics[key]

    frameGroup.create_dataset('fluidPosition', data = pernenialState['fluid']['positions'].detach().cpu().numpy())
    frameGroup.create_dataset('fluidVelocity', data = pernenialState['fluid']['velocities'].detach().cpu().numpy())
    frameGroup.create_dataset('fluidDensity', data = pernenialState['fluid']['densities'].detach().cpu().numpy() / config['fluid']['rho0'])

    if config['gravity']['active']:
        frameGroup.create_dataset('fluidGravity', data = pernenialState['fluid']['gravityAccel'].detach().cpu().numpy())

    frameGroup.create_dataset('fluidShiftAmount', data = perennialState['fluid']['shiftAmount'].detach().cpu().numpy())
    frameGroup.create_dataset('UID', data = perennialState['fluid']['index'].detach().cpu().numpy())
    if 'boundary' in perennialState and perennialState['boundary'] is not None:
        if dynamicBoundary:
            frameGroup.create_dataset('boundaryPosition', data = pernenialState['boundary']['positions'].detach().cpu().numpy())
        frameGroup.create_dataset('boundaryVelocity', data = pernenialState['boundary']['velocities'].detach().cpu().numpy())
        frameGroup.create_dataset('boundaryDensity', data = pernenialState['boundary']['densities'].detach().cpu().numpy() / config['fluid']['rho0'])
        if dynamicBoundary:
            frameGroup.create_dataset('boundaryUID', data = pernenialState['boundary']['index'].detach().cpu().numpy())
            frameGroup.create_dataset('boundaryNormal', data = pernenialState['boundary']['normals'].detach().cpu().numpy())
    


# print(config)

f, simulationDataGroup = initializeOutput(config, initialState)

priorState = None
for i in tqdm(range(256)):
    if 'shiftAmount' in perennialState['fluid']:
        del perennialState['fluid']['shiftAmount']
    perennialState, priorState, *updates = integrate(simulationStep, perennialState, config, previousStep= priorState)
    dx, _ = solveShifting(perennialState, config)
    perennialState['fluid']['shiftAmount'] = dx

    numNeighbors = perennialState['fluid']['neighborhood']['numNeighbors']
    perennialState['fluid']['positions'][numNeighbors > 5] += dx[numNeighbors > 5]

    
    if 'boundary' in initialState and initialState['boundary'] is not None:

        perennialState['boundaryGhost'] = buildBoundaryGhostParticles(perennialState, config)
        perennialState['boundary']['velocities'] = computeBoundaryVelocities(perennialState, config)
    # perennialState['boundary']['velocities'][perennialState['boundary']['positions'][:,1] > 0.5 - 1e-4,0] = 1

    if config['compute']['checkNaN']:
        checkNaN(perennialState['fluid']['positions'], 'positions')
        checkNaN(perennialState['fluid']['shiftAmount'], 'shiftAmount')
    
    perennialState['time'] += config['timestep']['dt']
    perennialState['timestep'] += 1

    Ek = 0.5 * perennialState['fluid']['masses'] * torch.sum(perennialState['fluid']['velocities']**2, dim = 1)
    state = perennialState
    nu = config['diffusion']['alpha'] * config['fluid']['cs'] * config['particle']['support'] / (2 * (config['domain']['dim'] +2))
    nu = config['diffusion']['nu'] if config['diffusion']['velocityScheme'] == 'deltaSPH_viscid' else nu
    dt_v = 0.125 * config['particle']['support']**2 / nu / config['kernel']['kernelScale']**2
    # acoustic timestep condition

    dt_c = config['timestep']['CFL'] * config['particle']['support'] / config['fluid']['cs'] / config['kernel']['kernelScale']    
    # print(dt_v, dt_c)

    # acceleration timestep condition
    if 'dudt' in state['fluid']: 
        dudt = state['fluid']['dudt']
        max_accel = torch.max(torch.linalg.norm(dudt[~torch.isnan(dudt)], dim = -1))
        dt_a = 0.25 * torch.sqrt(config['particle']['support'] / (max_accel + 1e-7)) / config['kernel']['kernelScale']
    else:
        dt_a = torch.tensor(config['timestep']['maxDt'], dtype = dt_v.dtype, device = dt_v.device)


    frameStatistics = computeStatistics(perennialState, particleState, config)
    
    if config['export']['active']:
        if perennialState['timestep'] % config['export']['interval'] == 0:
            exportFrame(perennialState, particleState, config, simulationDataGroup, dynamicBoundary = False)
            # writeFrame(simulationDataGroup, perennialState, priorState, frameStatistics, config)

    if 'boundary' in initialState and initialState['boundary'] is not None:
        fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"] + perennialState["boundary"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid + {perennialState["boundary"]["numParticles"]} boundary) E_k: {Ek.sum()}, Ratio: {Ek.sum() / Ek_0.sum() * 100 : .2f}\n$\\Delta t$ = {perennialState["dt"]} $\\Delta t_\\nu$ = {dt_v:.3g}, $\\Delta t_\\text{{CFL}}$ = {dt_c:.3g}, $\\Delta t_a$ = {dt_a:.3g} $\\bar\\rho$ = {perennialState["fluid"]["densities"].mean():.2f}')
    else:
        fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid) E_k: {Ek.sum()}, Ratio: {Ek.sum() / Ek_0.sum() * 100 : .2f}\n$\\Delta t$ = {perennialState["dt"]} $\\Delta t_\\nu$ = {dt_v:.3g}, $\\Delta t_\\text{{CFL}}$ = {dt_c:.3g}, $\\Delta t_a$ = {dt_a:.3g} $\\bar\\rho$ = {perennialState["fluid"]["densities"].mean():.2f}')


    # fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"] + perennialState["boundary"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid + {perennialState["boundary"]["numParticles"]} boundary) E_k: {Ek.sum()}, Ratio: {Ek.sum() / Ek_0.sum() * 100 : .2f}\n$\\Delta t$ = {perennialState["dt"]} $\\Delta t_\\nu$ = {dt_v}, $\\Delta t_\\text{{CFL}}$ = {dt_c}, $\\Delta t_a$ = {dt_a}')
    # print(dt_v, dt_c, dt_a)
    # fig.suptitle(f'Time: {perennialState["time"]:.2f} s, timestep: {perennialState["timestep"]}, dt: {perennialState["dt"]:.2e} s, particles: {perennialState["fluid"]["numParticles"] + perennialState["boundary"]["numParticles"]} ({perennialState["fluid"]["numParticles"]} fluid + {perennialState["boundary"]["numParticles"]} boundary) E_k: {Ek.sum()}, Ratio: {Ek.sum() / Ek_0.sum() * 100 : .2f}')
    if i % 16 == 0:
        # printState(perennialState)
        # print(f'Iteration {i}')
        visualizationState = prepVisualizationState(perennialState, config)
        updatePlot(uxPlot, visualizationState, 'velocities')
        updatePlot(uyPlot, visualizationState, 'velocities')
        updatePlot(indexPlot, visualizationState, 'index')
        # updatePlot(pressurePlot, visualizationState, 'densities')
        updatePlot(uPlot, visualizationState, 'velocities')
        fig.canvas.draw()
        fig.canvas.flush_events()
        exportPlot(perennialState, config, fig)

    for emitter in config['regions']:
        if emitter['type'] == 'inlet':
            processInlet(perennialState, emitter)
        if emitter['type'] == 'outlet':
            processOutlet(emitter, config, perennialState)
        if emitter['type'] == 'forcing':
            processForcing(emitter, config, perennialState)

f.close()

import subprocess
import shlex

outFile = config['plot']['namingScheme'] if config['plot']['namingScheme'] != 'timestep' else config["simulation"]["timestamp"]
if config['plot']['namingScheme'] == 'timestep':
    outFolder = f'{config["plot"]["exportPath"]}/{config["simulation"]["timestamp"]}/'
else:
    outFolder = f'{config["plot"]["exportPath"]}/{config["plot"]["namingScheme"]}/'

os.makedirs(outFolder, exist_ok = True)
# print('Creating video from  frames (frame count: {})'.format(len(os.listdir(outFolder))))
command = '/usr/bin/ffmpeg -loglevel warning -y -framerate 30 -f image2 -pattern_type glob -i '+ outFolder + '*.png -c:v libx264 -b:v 20M -r ' + str(config['plot']['exportFPS']) + ' ' + outFolder + 'output.mp4'
commandB = f'ffmpeg -loglevel warning -hide_banner -y -i {outFolder}output.mp4 -vf "fps={config["plot"]["exportFPS"]},scale={config["plot"]["gifScale"]}:-1:flags=lanczos,palettegen" output/palette.png'
commandC = f'ffmpeg -loglevel warning -hide_banner -y -i {outFolder}output.mp4 -i output/palette.png -filter_complex "fps={config["plot"]["exportFPS"]},scale={config["plot"]["gifScale"]}:-1:flags=lanczos[x];[x][1:v]paletteuse" {outFile}.gif'

subprocess.run(shlex.split(command))
subprocess.run(shlex.split(commandB))
subprocess.run(shlex.split(commandC))