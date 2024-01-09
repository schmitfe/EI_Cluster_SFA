import numpy as np
import pickle
import itertools
from pathlib import Path

def CreateParameterLists(fixed_parameters, variable_parameters, N_Jobs=1, OutputPath=None):
    """
    Create N_Jobs parameter lists from fixed_parameters and variable_parameters
    Each List schould be run on a single node
    :param fixed_parameters: dictionary with fixed parameters
    :param variable_parameters: dictionary with variable parameters
    :param N_Jobs: number of jobs
    :param OutputPath: path to output folder
    """
    if OutputPath is None:
        OutputPath = './'
    else:
        OutputPath = OutputPath + '/'

    #create output folder if it does not exist
    Path(OutputPath).mkdir(parents=True, exist_ok=True)

    #create all combinations of variable parameters
    Keys = list(variable_parameters.keys())
    #variable_parameters is a dictionary of lists
    Lengths = [len(variable_parameters[key]) for key in Keys]
    N_Combinations = np.prod(Lengths)
    Combinations = itertools.product(*[variable_parameters[key] for key in Keys])

    ParametersPerJob = int(np.ceil(N_Combinations / N_Jobs))
    #create a list of parameter combinations for each job
    for ii in range(N_Jobs):
        with open(OutputPath + 'Parameters_' + str(ii) + '.pkl', 'wb') as outfile:
            SubmitParameters = {'FixedParameter': fixed_parameters, 'VariableParameterNames': list(Keys)}
            # create a list of parameter combination for each job
            SubmitParameters['VariableParameterValues'] = []
            for jj in range(ParametersPerJob):
                try:
                    SubmitParameters['VariableParameterValues'].append(next(Combinations))
                except StopIteration:
                    break
            pickle.dump(SubmitParameters, outfile)
    return

if __name__ == '__main__':
    #create a dictionary with fixed parameters
    fixed_parameters = {'n_jobs': 1, 'N_E': 1200, 'N_I': 300, 'dt': 0.1, 'neuron_type': 'gif_psc_exp',
             'Q': 6, 'simtime': 10000., 'delta_I_xE': 0., 'delta_I_xI': 0., 'record_voltage': False,
             'warmup': 1000.,  'randseed': 100}
    #create a dictionary with variable parameters
    N_steps = 6
    variable_parameters = {'I_th_E': np.linspace(1.25, 3.0, N_steps), 'I_th_I': np.linspace(0.75, 2.0, N_steps),
                            'tau_stc': np.linspace(10., 500., N_steps), 'Q_adapt': np.linspace(0.0, 50.0, N_steps),
                            'Jep': np.linspace(1.0, 6.0, N_steps)}

    CreateParameterLists(fixed_parameters, variable_parameters, N_Jobs=40, OutputPath='/scratch/fschmi69/spontaneousSFA/Parameters')
