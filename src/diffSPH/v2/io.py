from diffSPH.v2.plotting import setupInitialPlot, updatePlots, postProcessPlot
from diffSPH.kernels import KernelWrapper
import h5py
import torch
import os


def createOutputFile(config):
    outFile = config['plot']['namingScheme'] if config['plot']['namingScheme'] != 'timestep' else config["simulation"]["timestamp"]
    os.makedirs(config['export']['exportPath'], exist_ok = True)
    outputFile = f'{config["export"]["exportPath"]}/{outFile}.hdf5'

    # f.close()
    f = h5py.File(outputFile, 'w')

    f.attrs['support'] = config['particle']['support'].detach().cpu().item()
    f.attrs['targetNeighbors'] = config['kernel']['targetNeighbors'].detach().cpu().item()
    f.attrs['restDensity'] = config['fluid']['rho0']
    f.attrs['radius'] = config['particle']['dx'].detach().cpu().item()
    f.attrs['area'] = config['particle']['dx'].detach().cpu().item()**2
    if not config['gravity']['active']:
        f.attrs['fluidGravity'] = [0, 0] 
    # f.attrs['fluidGravity'] = [0, 0]

    configGroup = f.create_group('config')
    for key in config:
        configGroup.create_group(key)
        for subkey in config[key]:
            # print(f'Writing {key}/{subkey}')
            val = config[key][subkey]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            if isinstance(val, torch.dtype) or \
                isinstance(val, torch.device) or \
                hasattr(val, '__call__') or \
                isinstance(val, KernelWrapper):
                continue
            if isinstance(val, dict):
                # print(f'Skipping dict {key}/{subkey}')
                continue
                for subsubkey in val:
                    print(f'Writing {key}/{subkey}/{subsubkey}')
                    configGroup[key].create_dataset(subkey, data=val[subsubkey])
                continue

            # print(f'Writing {key}/{subkey} ({config[key][subkey]})')
            configGroup[key].attrs[subkey] = val

    # f.close()
    # boundaryGroup = f.create_group('boundaryInformation')
    # boundaryGroup.create_dataset('boundaryPosition', [])
    # boundaryGroup.create_dataset('boundaryNormal', [])
    # boundaryGroup.create_dataset('boundaryArea', [])
    # boundaryGroup.create_dataset('boundaryVelocity', [])
    simulationDataGroup = f.create_group('simulationExport')

    return f, simulationDataGroup
def writeFrame(simulationDataGroup, perennialState, priorState, frameStatistics, config):
    frameGroup = simulationDataGroup.create_group(f'{perennialState["timestep"]:05d}')

    for key in frameStatistics:
        frameGroup.attrs[key] = frameStatistics[key]

    frameGroup.create_dataset('fluidPosition', data = priorState['fluid']['positions'].detach().cpu().numpy())
    frameGroup.create_dataset('fluidVelocity', data = priorState['fluid']['velocities'].detach().cpu().numpy())
    frameGroup.create_dataset('fluidDensity', data = priorState['fluid']['densities'].detach().cpu().numpy() / config['fluid']['rho0'])
    if config['gravity']['active']:
        frameGroup.create_dataset('fluidGravity', data = priorState['fluid']['gravityAccel'].detach().cpu().numpy())

    # frameGroup.create_dataset('fluidEk', data = perennialState['fluidEks'].detach().cpu().numpy())

    # frameGroup.create_dataset('fluid_drhodt', data = perennialState['fluid_drhodt'].detach().cpu().numpy())
    # frameGroup.create_dataset('fluid_dudt', data = perennialState['fluid_dudt'].detach().cpu().numpy())
    # frameGroup.create_dataset('fluid_dxdt', data = perennialState['fluid_dxdt'].detach().cpu().numpy())
    frameGroup.create_dataset('fluidShiftAmount', data = perennialState['fluid']['shiftAmount'].detach().cpu().numpy())
    frameGroup.create_dataset('UID', data = perennialState['fluid']['index'].detach().cpu().numpy())
    
    # frameGroup.create_dataset('boundaryDensity', [])
