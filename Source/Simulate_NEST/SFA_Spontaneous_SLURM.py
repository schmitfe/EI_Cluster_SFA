import numpy as np
import pickle
import sys
import os
from pathlib import Path
import hashlib

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelNEST
from Helper import ExtractMeasurement
from Helper import spiketools
import itertools
import time




if __name__ == '__main__':
    #Start = time.time()
    JobID = os.environ.get('SLURM_JOB_ID', '0')
    ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    CPUcount = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    #get git hash
    gitHash = os.popen('git rev-parse HEAD').read().strip()

    #get enviroment variables for simulation parmaeters
    randseed = int(os.environ.get('randseed', str(np.random.randint(1000000))))   # set this to fixed value across nodes
    # Jep, jip_ratio, I_th_E, I_th_I, N_E, tau_stc, q_stc
    jep = float(os.environ.get('Jep', str(default.jep)))                # Tune this parameter
    jip_ratio = float(os.environ.get('jip_ratio', '0.75'))
    I_th_E = float(os.environ.get('I_th_E', str(default.I_th_E)))       # Tune this parameter
    I_th_I = float(os.environ.get('I_th_I', str(default.I_th_I)))       # Tune this parameter
    N_E = int(os.environ.get('N_E', str(default.N_E)))
    N_I = int(os.environ.get('N_I', str(default.N_I)))
    tau_stc = float(os.environ.get('tau_stc', str(default.tau_stc)))    # Tune this parameter
    # if Q_adapt is in the environment variables, use it
    if 'Q_adapt' in os.environ:
        Q_adapt = float(os.environ['Q_adapt'])                          # Tune this parameter
        q_stc = Q_adapt/tau_stc

    else:
        q_stc = float(os.environ.get('q_stc', str(default.q_stc)))
    #get enviroment variables for output_path
    output_path = os.environ.get('output_path', '../output/')
    #get enviroment variables for simulation protocol
    #Trials per direction, ITI, Preperatory duration, Stimulus duration, Stimulus intensity, Stimulus type
    Trials = int(os.environ.get('Trials', '20'))
    ITI = [float(os.environ.get('ITI_min', '200')), float(os.environ.get('ITI_max', '1000'))]
    TrialDuration = float(os.environ.get('TrialDuration', '400'))


    #check if N_I is a multiple of Q and N_E is 4 times N_I
    if N_I % default.Q != 0:
        raise ValueError('N_I must be a multiple of Q')
    if N_E != 4 * N_I:
        raise ValueError('N_E must be 4 times N_I')


    #calculate inhibitory weight
    jip = 1. + (jep - 1) * jip_ratio

    #test if output path ends with a directory seperator
    if output_path.endswith(os.path.sep):
        #create a short name for the output file based on the parameters hash
        #output_file = 'jep_{}_jip_{}_IthE_{}_IthI_{}_NE_{}_NI_{}_tau_{}_q_{}_seed_{}'\
        #    .format(jep, jip, I_th_E, I_th_I, N_E, N_I, tau_stc, q_stc, randseed)
        output_file = 'job_{}_array_{}'.format(JobID, ArrayID)
        #hash the output file name and encode it to utf-8 string
        #output_file = hashlib.sha1(output_file.encode('utf-8')).hexdigest()
        #add the file extension
        output_file = output_file + '.pkl'
        #join the output path and the output file
        output_path = os.path.join(output_path, output_file)

    #create output directory if it does not exist
    try:
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except:
        pass

    #test if output file already exists
    #if os.path.isfile(output_path):
    #    pass
    #else:
    #    FileCreated = False
        #raise ValueError('Output file already exists')


    print(output_path)

    params= {'n_jobs': CPUcount, 'N_E': N_E, 'N_I': N_I, 'dt': 0.1, 'neuron_type': 'gif_psc_exp',
             'Q': 6, 'jplus': np.array([[jep, jip], [jip, jip]]), 'I_th_E': I_th_E, 'I_th_I': I_th_I,
             'warmup': 1000., 'jep': jep, 'jip_ratio': jip_ratio, 'tau_stc': tau_stc, 'Q_adapt': Q_adapt, 'q_stc': q_stc, 'randseed': randseed}

    # calculate simulation duration, start and end times of trials
    np.random.seed(int(ArrayID))

    ITIs=np.random.randint(ITI[0], ITI[1], size=Trials)
    # first ITI is placed after warmup, before first trial

    # Trials are TrialDuration ms long and first trial starts after warmup
    Trial_end = np.cumsum(ITIs) + np.cumsum(np.ones(Trials)*TrialDuration)
    Trial_start = Trial_end - TrialDuration

    params['simtime'] = Trial_end[-1] + 200.

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST(default, params)
    EI_Network.setup_network()
    EI_Network.simulate()

    # Analyze data, we are interested in the average firing rate of the excitatory and inhibitory population,
    # the average Fano factor of excitatory and inhibitory population and the average CV2 of the excitatory and
    # inhibitory population. We can sample 10 % of the neurons to get a good estimate of the population average.
    # But we need to make sure that we sample at least 10 neurons and sample the same neuron across jobs.

    np.random.seed(1)
    # get the neuron ids of the excitatory and inhibitory population
    N_E = params['N_E']
    N_I = params['N_I']
   # sample 10 % of the neurons
    N_E_sample = max(int(N_E / 10), 10)
    N_I_sample = max(int(N_I / 10), 10)

    # sample neurons
    E_sample = np.random.choice(N_E, N_E_sample, replace=False)

    I_sample = np.random.choice(N_I, N_I_sample, replace=False)+params['N_E']

    # get the spike trains of the sampled neurons
    Spikes = EI_Network.get_recordings()
    #print(EI_Network.get_firing_rates(Spikes))
    E_Spikes = ExtractMeasurement.filterSpikes(Spikes, IDs=E_sample)
    I_Spikes = ExtractMeasurement.filterSpikes(Spikes, IDs=I_sample)
    # if there are no spikes, we can not estimate the Fano factor and CV2, so we return nan


    del Spikes
    # add 3rd row to spike trains with the trial number
    E_Spikes = ExtractMeasurement.CutTrials(E_Spikes, Trial_start, Trial_end)
    I_Spikes = ExtractMeasurement.CutTrials(I_Spikes, Trial_start, Trial_end)

    # get the firing rates per neuron and trial, create list of tuples with (neuron_id, trial_id) and get counts of
    # unique tuples to get the number of spikes per neuron and trial
    E_Tuples = [(Spikes[1], Spikes[2]) for Spikes in E_Spikes.T]
    Values_E , E_SpikeCounts = np.unique(E_Tuples, return_counts=True, axis=0)
    I_Tuples = [(Spikes[1], Spikes[2]) for Spikes in I_Spikes.T]
    Values_I, I_SpikeCounts = np.unique(I_Tuples, return_counts=True, axis=0)
    Values_E = np.array(Values_E).T
    Values_I = np.array(Values_I).T
    # add the missing combinations of neurons and trials with 0 spikes
    PossibleValues_E = set(itertools.product(E_sample, range(Trials)))
    PossibleValues_I = set(itertools.product(I_sample, range(Trials)))
    MissingValues_E = PossibleValues_E - set(map(tuple, Values_E.T))
    MissingValues_I = PossibleValues_I - set(map(tuple, Values_I.T))
    MissingValues_E = np.array(list(map(list, list(MissingValues_E))))
    MissingValues_I = np.array(list(map(list, list(MissingValues_I))))
    if len(Values_E) == 0:
        Values_E = MissingValues_E.T
        E_SpikeCounts = np.zeros(len(MissingValues_E))
    else:
        Values_E = np.concatenate((Values_E, MissingValues_E.T), axis=1)
        E_SpikeCounts = np.concatenate((E_SpikeCounts, np.zeros(len(MissingValues_E))), axis=0)

    if len(Values_I) == 0:
        Values_I = MissingValues_I.T
        I_SpikeCounts = np.zeros(len(MissingValues_I))
    else:
        Values_I = np.concatenate((Values_I, MissingValues_I.T), axis=1)
        I_SpikeCounts = np.concatenate((I_SpikeCounts, np.zeros(len(MissingValues_I))), axis=0)

    # sort the arrays - not necessary but makes it easier to read
    E_SpikeCounts = E_SpikeCounts[np.argsort(Values_E[0])]
    Values_E = Values_E[:, np.argsort(Values_E[0])]
    I_SpikeCounts = I_SpikeCounts[np.argsort(Values_I[0])]
    Values_I = Values_I[:, np.argsort(Values_I[0])]

    # estimate firing rates per neuron
    E_FiringRate = np.mean(E_SpikeCounts / (TrialDuration / 1000.))
    I_FiringRate = np.mean(I_SpikeCounts / (TrialDuration / 1000.))
    #print("Firing Rates (E,I):  ", E_FiringRate, I_FiringRate)
    # estimate Fano factor per neuron
    E_FanoFactor = np.zeros_like(E_sample, dtype=float)
    I_FanoFactor = np.zeros_like(I_sample, dtype=float)
    for ii in range(N_E_sample):
        NeuronCounts = E_SpikeCounts[Values_E[0] == E_sample[ii]]
        E_FanoFactor[ii] = np.var(NeuronCounts) / np.mean(NeuronCounts)
    for ii in range(N_I_sample):
        NeuronCounts = I_SpikeCounts[Values_I[0] == I_sample[ii]]
        I_FanoFactor[ii] = np.var(NeuronCounts) / np.mean(NeuronCounts)
    #print("Fano Factors (E,I):  ", np.nanmean(E_FanoFactor), np.nanmean(I_FanoFactor))
    #print("Fano Factors (E):  ", E_FanoFactor)
    #print("-------------------------------------")
    #print("Fano Factors (I):  ", I_FanoFactor)

    #print("-------------------------------------")
    # estimate CV2 per neuron
    CV2_E = np.zeros(N_E_sample, dtype=float)
    for ii, Unit in enumerate(E_sample):
        SpikesUnit = E_Spikes[:, E_Spikes[1] == Unit]
        CV2_E[ii] = spiketools.cv_two(SpikesUnit[[0,2], :], min_vals=10)
    CV2_I = np.zeros(N_I_sample, dtype=float)
    for ii, Unit in enumerate(I_sample):
        SpikesUnit = I_Spikes[:, I_Spikes[1] == Unit]
        CV2_I[ii] = spiketools.cv_two(SpikesUnit[[0,2], :], min_vals=10)

    #print("CV2 (E):  ", np.nanmean(CV2_E))
    #print("-------------------------------------")
    #print("CV2 (E):  ", CV2_E)
    #print("-------------------------------------")




    #print("Parameters:  ", EI_Network.get_parameter())




    #with open(output_path, 'wb') as outfile:
    #    pickle.dump(EI_Network.get_recordings(), outfile)
    #    pickle.dump(EI_Network.get_parameter(), outfile)
    #    pickle.dump(StimulusDict, outfile)
    #    pickle.dump({'gitHash': gitHash, 'JobID': JobID, 'ArrayID': ArrayID}, outfile)
    EI_Network.clean_up()
    ResultDict={'E_FiringRate': np.nanmean(E_FiringRate), 'I_FiringRate': np.nanmean(I_FiringRate), 'E_FanoFactor': np.nanmean(E_FanoFactor),
                     'I_FanoFactor': np.nanmean(I_FanoFactor), 'CV2_E': np.nanmean(CV2_E), 'CV2_I': np.nanmean(CV2_I),
                     **EI_Network.get_parameter()}
    #End = time.time()
    if os.path.isfile(output_path):
        with open(output_path, 'rb') as infile:
            Data = pickle.load(infile)
        Data.append(ResultDict)
    else:
        Data = [ResultDict]

    with open(output_path, 'wb') as outfile:
        pickle.dump(Data, outfile)

    # return exit code 0 if everything went fine
    sys.exit(0)
