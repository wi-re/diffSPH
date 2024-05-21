import torch
from tqdm.autonotebook import tqdm
from diffSPH.v2.plotting import setupInitialPlot, updatePlots, postProcessPlot
from diffSPH.v2.util import computeStatistics
from diffSPH.v2.modules.integration import integrate
from diffSPH.v2.io import createOutputFile, writeFrame
from diffSPH.v2.modules.shifting import solveShifting
from diffSPH.v2.modules.timestep import computeTimestep
import copy

def setupSimulation(particleState, config, stepLimit = 1000, timeLimit = -1, copyState = False):
    if copyState:
        perennialState = copy.deepcopy(particleState)
    fig, axis, plotStates = setupInitialPlot(particleState, particleState, config)
    priorState = None
    # updatePlots(perennialState, particleState, config, plotStates, fig, axis)
    
    pbar = tqdm(total=timeLimit if timeLimit > 0 else stepLimit, bar_format = "{desc}: {percentage:.4f}%|{bar}| {n:.4g}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}{postfix}", leave=False)

    stats = []
    if copyState:
        return perennialState, fig, axis, plotStates, priorState, pbar, stats
    return particleState, fig, axis, plotStates, priorState, pbar, stats

def runSimulation(fig, axis, simulationStep, plotStates, priorState, pbar, stats, perennialState, particleState, config, stepLimit = 1000, timeLimit = -1, callBack = None):
    # for i in tqdm(range(1000)):
    frameStatistics = computeStatistics(perennialState, particleState, config)
    stats.append(frameStatistics)
    if perennialState['timestep'] % config['plot']['updateInterval'] == 0:
        updatePlots(perennialState, particleState, config, plotStates, fig, axis)
    if config['export']['active']:
        f, simulationDataGroup = createOutputFile(config)
        
    lastUpdate = perennialState['time']
    while(True):
        # print('Integrating')
        perennialState, priorState, *updates = integrate(simulationStep, perennialState, config, previousStep= priorState)
        # Particle shifting
        # print('Solving shifting')
        dx, _ = solveShifting(perennialState, config)
        perennialState['fluid']['shiftAmount'] = dx
        perennialState['fluid']['positions'] += dx
        # print('Computing timestep')
        # Frame done, update state for next timestep
        perennialState['dt'] = config['timestep']['dt']
        perennialState['fluid']['Eks'] = (0.5 * perennialState['fluid']['areas'] * perennialState['fluid']['densities'] * torch.linalg.norm(perennialState['fluid']['velocities'], dim = -1)**2)
        perennialState['fluid']['E_k'] = perennialState['fluid']['Eks'].sum().detach().cpu().item()
        frameStatistics = computeStatistics(perennialState, particleState, config)
        
        if config['export']['active']:
            if perennialState['timestep'] % config['export']['interval'] == 0:
                writeFrame(simulationDataGroup, perennialState, priorState, frameStatistics, config)

        
        perennialState['time'] += config['timestep']['dt']
        perennialState['timestep'] += 1

        config['timestep']['dt'] = computeTimestep(perennialState, config)
        time = perennialState['time']
        dt = config['timestep']['dt']

        if timeLimit > 0:
            pbar.n = time.detach().cpu().item() if isinstance(time, torch.Tensor) else time
            pbar.refresh()
        else: 
            pbar.n = perennialState['timestep']
            pbar.refresh()
        if perennialState['time'] > timeLimit and timeLimit > 0:
            break
        if perennialState['timestep'] > stepLimit and stepLimit > 0:
            break



        ttime = perennialState['time'] if not isinstance(perennialState['time'], torch.Tensor) else perennialState['time'].cpu().item()
        stats.append(frameStatistics)
        if callBack is not None:
            callBack(perennialState, particleState, config, plotStates, fig, axis, frameStatistics)

        if config['plot']['fps'] > 0:
            if perennialState['time'] > lastUpdate + 1 / config['plot']['fps']:
                lastUpdate = ttime
                updatePlots(perennialState, particleState, config, plotStates, fig, axis, title = callBack)
        else:
            if perennialState['timestep'] % config['plot']['updateInterval'] == 0:
                updatePlots(perennialState, particleState, config, plotStates, fig, axis, title = callBack)

    pbar.close()
    if config['export']['active']:
        f.close()

    postProcessPlot(config)
    return stats, perennialState