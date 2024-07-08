import pickle
import numpy as np
import pandas as pd
import glob
import sys
import os
import itertools
from pathlib import Path


def load_jobs(path):
    """ Load jobs from path.
    Parameters
    ----------
    path : str
        Path to job files.
    filename : str
        Filename of job files.
    Returns
    -------
    df : pandas.DataFrame
        Dataframe with jobs.
    """



    # create pandas dataframe from jobs
    df = pd.read_pickle(path)
    df['PS_amplitude'] = df['multi_stim_amps']
    df['RS_amplitude'] = df['multi_stim_amps']
    # drop duplicates (E_FiringRate and I_FiringRate, E_FanoFactor and I_FanoFactor are enought to compare)

    return df

def filter_jobs(df, filter_dict, verbose=False):
        """ Filter jobs.
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with jobs.
        filter_dict : dict
            Dictionary with filter parameters. Keys are column names, Tuples are lower and upper bounds.
        Returns
        -------
        df : pandas.DataFrame
            Dataframe with filtered jobs.
        """

        # filter jobs
        if verbose:
            FilterIndex = np.ones((len(df), len(filter_dict)), dtype=bool)
            for ii, (key, value) in enumerate(filter_dict.items()):
                FilterIndex[:, ii] = (df[key] >= value[0]) & (df[key] <= value[1])
                print(key, value, np.sum(FilterIndex[:, ii]))
            FilterIndex = np.all(FilterIndex, axis=1)
            print('Total number of jobs: ', np.sum(FilterIndex))
            df = df[FilterIndex]
        else:
            for key, value in filter_dict.items():
                df = df[(df[key] >= value[0]) & (df[key] <= value[1])]

        return df

def remove_measurements(df):
    df = df.drop(columns=['FR_E_stim', 'FR_E_nonstim', 'Time_FR', 'FanoFactor_E_stim', 'FanoFactor_E_nonstim', 'Time_Fano', 'CV2_E_stim', 'CV2_E_nonstim','Time_CV2','average_FR', 'average_Fano', 'average_CV2', 'Loss_FR', 'Loss_Fano', 'Loss_CV2']).reset_index(drop=True)
    return df

def remove_other_columns(df):
    df = df.drop(columns=['record_voltage',
                          'record_from', 'recording_interval', 'return_weights',
                          'stim_clusters', 'multi_stim_clusters', 'multi_stim_times', 'multi_stim_amps', 'stim_amp', 'stim_starts',
                          'stim_ends', 'simtime', 'warmup', 'js', 'jplus', 'RS_amplitude', 'PS_amplitude'], errors='ignore').reset_index(drop=True)
    return df

def unique_series(input_list):
    # test if series contains elements which are not floats or ints
    if not type(input_list[0]) in [np.float64, np.int64, float, int, str, dict, list]:
        dimension=input_list[0].ndim
        unique_values=np.squeeze(np.unique(np.stack(input_list,dimension),axis=dimension))

        if unique_values.ndim == dimension:
            return True
        else:
            return False
    elif type(input_list[0]) in [dict]:
        # test if all dictionaries are the same
        test_dict = input_list[0]
        for dict_ in input_list:
            if dict_ != test_dict:
                return False
        # if all dictionaries are the same return True
        return True
    elif type(input_list[0]) in [list]:
        # test if all lists are the same
        test_list = input_list[0]
        for list_ in input_list:
            if ~all(x == y for x, y in zip(list_, test_list)):
                return False
        # if all lists are the same return True
        return True
    else:
        return np.unique(input_list).size==1


def CreateParameterLists(df, variable_parameters, N_Jobs=1, OutputPath=None):
    """
    Create N_Jobs parameter lists from fixed_parameters and variable_parameters
    Each List schould be run on a single node
    :param df: dataframe with approved samples from previous search
    :param variable_parameters: dictionary with variable parameters
    :param N_Jobs: number of jobs
    :param OutputPath: path to output folder
    """
    if OutputPath is None:
        OutputPath = './'
    else:
        OutputPath = OutputPath + '/'

    FixedParameters = {}
    VariableParameterNames = []
    for key in df.columns:
        if unique_series(df[key]):
            # only add if not a variable parameter
            if key not in variable_parameters.keys():
                FixedParameters[key] = df[key].iloc[0]
        else:
            VariableParameterNames.append(key)

    #create output folder if it does not exist
    Path(OutputPath).mkdir(parents=True, exist_ok=True)

    # create all combinations of (row) indices and variable parameters
    Keys = list(variable_parameters.keys())
    #variable_parameters is a dictionary of lists
    Lengths = [len(variable_parameters[key]) for key in Keys]
    N_Combinations = np.prod(Lengths+ [len(df)])
    Combinations = itertools.product(*[variable_parameters[key] for key in Keys], range(len(df)))

    print('Total number of combinations: ', N_Combinations)

    ParametersPerJob = int(np.ceil(N_Combinations / N_Jobs))
    #create a list of parameter combinations for each job
    for ii in range(N_Jobs):
        with open(OutputPath + 'Parameters_' + str(ii) + '.pkl', 'wb') as outfile:
            SubmitParameters = {'FixedParameter': FixedParameters, 'VariableParameterNames': list(Keys)+VariableParameterNames}
            # create a list of parameter combination for each job
            SubmitParameters['VariableParameterValues'] = []
            for jj in range(ParametersPerJob):
                try:
                    NextCombination = next(Combinations)
                    ExpendCombination = NextCombination[:-1]
                    # append only values from dataframe row specified in VariableParameterNames
                    ExpendCombination += tuple(df.iloc[NextCombination[-1]][VariableParameterNames])
                    SubmitParameters['VariableParameterValues'].append(ExpendCombination)
                except StopIteration:
                    break
            pickle.dump(SubmitParameters, outfile)
    return


if __name__=='__main__':

    #path = '/home/fschmitt/Documents/git/EI_Cluster_SFA/Source/output/pareto_front.pkl'
    path = '/scratch/fschmi69/stimulationSFA/Vahid_params.pkl'
        #'/home/fschmitt/Documents/git/EI_Cluster_SFA/Source/Simulate_NEST/SLURMSubmission/Vahid_params.pkl'

    AdditionalVariable_parameters = {'Condition': [1,2,3],
                                     'jep': np.linspace(3.2, 3.8, 7, endpoint=True),
                                     'PS_amplitude': np.linspace(0.05, 0.25, 9, endpoint=True),
                                     'Q_adapt': np.linspace(0.5, 4.5, 5, endpoint=True),
                                     'tau_stc': np.linspace(90., 120., 4, endpoint=True),
                                     'jip_ratio': np.linspace(0.71, 0.75, 5, endpoint=True)
                                     }


    # load jobs
    df = load_jobs(path)
    # use only first row
    df = df.iloc[:1]
    TotalSamples = len(df)

    df = remove_measurements(df)

    # drop columns which are not needed and related to the stimulus
    df = remove_other_columns(df)

    # create parameter lists
    CreateParameterLists(df, AdditionalVariable_parameters, N_Jobs=2000, OutputPath='/scratch/fschmi69/stimulationSFA/Parameters') #'/home/fschmitt/Documents/git/EI_Cluster_SFA/Source/Simulate_NEST/SLURMSubmission/Parameters'

