import torch


def stiffTaitEOS(simulationState, config):
    density = simulationState['fluidDensities']    
    reference_density = config['fluid']['rho0']
    sound_speed = config['fluid']['cs']
    exponent = config['EOS']['polytropicExponent']

    return reference_density * sound_speed**2 / exponent * ((density / reference_density)**exponent - 1)

def TaitEOS(simulationState, config):
    density = simulationState['fluidDensities']
    reference_density = config['fluid']['rho0']
    kappa = config['EOS']['kappa']

    return kappa * (density - reference_density)


def idealGasEOS(simulationState, config):
    density = simulationState['fluidDensities']
    temperature = simulationState['fluidTemperatures']
    gas_constant = config['gas_constant'] 

    return density * gas_constant * temperature / config['EOS']['molarMass']

def isoThermalEOS(simulationState, config):
    density = simulationState['fluidDensities']
    reference_density = config['fluid']['rho0']
    sound_speed = config['fluid']['cs']

    return sound_speed**2 * (density - reference_density)

def polytropicEOS(simulationState, config):
    density = simulationState['fluidDensities']
    exponent = config['EOS']['polytropicExponent']

    return config['EOS']['kappa'] * (density)**exponent

def murnaghanEOS(simulationState, config):
    density = simulationState['fluidDensities']
    reference_density = config['fluid']['rho0']
    exponent = config['EOS']['polytropicExponent']

    return config['EOS']['kappa'] / exponent * ((density / reference_density)**exponent - 1)

from torch.profiler import record_function
def computeEOS(simulationState, config):
    with record_function("EOS"):
        if config['EOS']['type'] == 'stiffTait':
            return stiffTaitEOS(simulationState, config)
        elif config['EOS']['type'] == 'Tait':
            return TaitEOS(simulationState, config)
        elif config['EOS']['type'] == 'idealGas':
            return idealGasEOS(simulationState, config)
        elif config['EOS']['type'] == 'isoThermal':
            return isoThermalEOS(simulationState, config)
        elif config['EOS']['type'] == 'polytropic':
            return polytropicEOS(simulationState, config)
        elif config['EOS']['type'] == 'murnaghan':
            return murnaghanEOS(simulationState, config)
        else:
            raise ValueError('EOS type not recognized')

    
from diffSPH.parameter import Parameter
def getParameters():
    return [
        Parameter('EOS', 'gasConstant', float, 8.14, required = False,export = False, hint = 'Gas constant for the ideal gas equation of state'),
        Parameter('EOS', 'molarMass', float, 0.02897, required = False,export = False, hint = 'Molar mass of the gas'),
        Parameter('EOS', 'polytropicExponent', float, 7, required = False,export = False, hint = 'Exponent for the Tait equation of state'),
        Parameter('EOS', 'kappa', float, 1.3, required = False,export = False, hint = 'Kappa for the less stiff Tait equation of state'),
        Parameter('EOS', 'type', str, 'isoThermal', required = False,export = False, hint = 'Type of equation of state')
    ]