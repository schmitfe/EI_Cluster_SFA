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

#generate stimulus protocol pseudo ranomly for each direction and condition
def generateStimulusProtocol_Testing(Q, TrialsPerDirection, ITI, PreperatoryDuration, StimulusDuration, PrepIntensity, StimulusIntensity, Directions, Conditions):
    #generate boolean array for stimulus protocol with shape 2*TrialPerDirection*length(Directions)*length(Conditions) x Q
    #generate pseudorandom list of directions, conditions with TrialsPerDirection*length(Directions)*length(Conditions) elements

    Protocol = []
    for i in range(TrialsPerDirection):
        for direction in Directions:
            for condition in Conditions:
                Protocol.append((direction, condition))
    np.random.shuffle(Protocol)
    #generate stimulus protocol
    StimulusProtocol = np.zeros((2*TrialsPerDirection*len(Directions)*len(Conditions), Q), dtype=bool)
    #use even row indexes for preperatory stimulus and odd row indexes for stimulus
    for ii, (direction, condition) in enumerate(Protocol):
        StimulusProtocol[2*ii+1, direction] = True
        #condition 1: preperatory stimulus same as stimulus
        for jj in range(direction//condition*condition, (direction//condition+1)*condition):
            #modulo operator to get correct index for last condition
            StimulusProtocol[2*ii, jj%Q] = True

    #generate list of Q lists for StimTimes and StimAmplitudes
    StimTimes = [[0.0] for i in range(Q)]
    StimAmplitudes = [[0.0] for i in range(Q)]

    #Marker with 3 columns: 0. time, 1. direction, 2. condition
    Marker = np.zeros((TrialsPerDirection*len(Directions)*len(Conditions),3))

    # if ITI is a single value, use it for all trials, otherwise use for each trial corresponding value
    if type(ITI) == float or type(ITI) == int:
        ITI = [ITI for i in range(TrialsPerDirection*len(Directions)*len(Conditions)+1)]
    # else check for correct length
    elif len(ITI) != TrialsPerDirection*len(Directions)*len(Conditions)+1:
        raise ValueError('ITI must have length TrialsPerDirection*len(Directions)*len(Conditions)+1')

    #iterate through stimulus protocol and generate stimulus times and amplitudes
    time=ITI[0]
    for ii in range(StimulusProtocol.shape[0]):
        #get stimulus times and amplitudes
        if ii%2 == 0:
            #preperatory stimulus
            #insert for each cluster which is marked as True in the stimulus protocol the stimulus amplitude of PrepIntensity
            for jj in range(Q):
                if StimulusProtocol[ii, jj]:
                    StimAmplitudes[jj].append(PrepIntensity)
                    StimTimes[jj].append(time)
            Marker[ii//2,0] = time      #Time of preperatory stimulus
            Marker[ii//2,1] = np.argwhere(StimulusProtocol[ii+1, :])[0][0]
            Marker[ii//2,2] = np.sum(StimulusProtocol[ii, :])
            time+=PreperatoryDuration

        else:
            #stimulus
            # insert for each cluster which is marked as True in the stimulus protocol the stimulus amplitude of
            # StimulusIntensity and if it was active during preperation but not during stimulus reset to 0
            for jj in range(Q):
                if StimulusProtocol[ii, jj]:
                    StimAmplitudes[jj].append(StimulusIntensity)
                    StimTimes[jj].append(time)
                    # reset to 0 after stimulus
                    StimAmplitudes[jj].append(0.0)
                    StimTimes[jj].append(time+StimulusDuration)
                elif StimulusProtocol[ii-1, jj]:
                    StimAmplitudes[jj].append(0.0)
                    StimTimes[jj].append(time)
            time+=StimulusDuration+ITI[(ii-1)//2+1]
    #convert to numpy arrays
    for jj in range(Q):
        StimTimes[jj] = np.array(StimTimes[jj])
        StimAmplitudes[jj] = np.array(StimAmplitudes[jj])
    Clusters= [[ii] for ii in range(Q)]


    # The list StimAmplitudes can contain chains of the same amplitude, we want to remove them and only keep the
    # first, we also need to remove the corresponding times - only apply in condition 1
    if condition == 1:
        # for jj in range(Q):
        #     ii = 0
        #     while ii < len(StimAmplitudes[jj])-1:
        #         if StimAmplitudes[jj][ii] == StimAmplitudes[jj][ii+1]:
        #             StimAmplitudes[jj] = np.delete(StimAmplitudes[jj], ii+1)
        #             StimTimes[jj] = np.delete(StimTimes[jj], ii+1)
        #         else:
        #             ii += 1


        for jj in range(Q):
            # Find where the next element is different
            diff = np.diff(StimAmplitudes[jj], prepend=StimAmplitudes[jj][0]-1) != 0

            # Index the original arrays with this
            StimAmplitudes[jj] = StimAmplitudes[jj][diff]
            StimTimes[jj] = StimTimes[jj][diff]


    StimulusDict= {'Marker': Marker, 'ITI': ITI, 'PreperatoryDuration': PreperatoryDuration,
                   'StimulusDuration': StimulusDuration, 'PrepIntensity': PrepIntensity,
                   'StimulusIntensity': StimulusIntensity}
    return StimulusDict, Clusters, StimTimes, StimAmplitudes

def calculate_weighted_L2_distance(x, y, w):
    return np.sqrt((w*(x-y)).t@(w*(x-y)))

def calculate_mean_measure(stim_cluster_signal, non_stim_signal, Q):
    return (1/Q)*stim_cluster_signal+(Q-1)/(Q)*non_stim_signal

def calculate_weighted_L2_distance_Timeshift(x, y, w, shift):
    if shift > 0:
        return calculate_weighted_L2_distance(x[:-shift], y[shift:], w[:-shift])
    else:
        return calculate_weighted_L2_distance(x[shift:], y[:-shift], w[shift:])


def check_same_vectors(list_of_arrays):
    first_array = list_of_arrays[0]
    return all(np.array_equal(first_array, arr) for arr in list_of_arrays[1:])

if __name__ == '__main__':
    #Start = time.time()
    JobID = os.environ.get('SLURM_JOB_ID', '0')
    ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    CPUcount = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    #get git hash
    gitHash = os.popen('git rev-parse HEAD').read().strip()

    #get enviroment variables for simulation parmaeters
    randseed = int(os.environ.get('randseed', str(np.random.randint(1000000))))   # set this to fixed value across nodes
    # jep, jip_ratio, I_th_E, I_th_I, N_E, tau_stc, q_stc
    # we use small case for jep -> error in initial spontaneous search
    jep = float(os.environ.get('jep', str(default.jep)))                # Tune this parameter
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
        Q_adapt = q_stc*tau_stc
    #get enviroment variables for output_path
    output_path = os.environ.get('output_path', '../output/')
    #get enviroment variables for simulation protocol
    #Trials per direction, ITI, Preperatory duration, Stimulus duration, Stimulus intensity, Stimulus type
    Trials = int(os.environ.get('Trials', '40'))
    ITI = [float(os.environ.get('ITI_min', '1000')), float(os.environ.get('ITI_max', '2000'))]
    #TrialDuration = float(os.environ.get('TrialDuration', '400'))

    # generate stimulus protocol
    PS_amplitude = float(os.environ.get('PS_amplitude', '0.3'))
    # parse condition which is a string of digits which should be converted to a list of integers
    Condition = os.environ.get('Condition', '1')
    Condition = [int(i) for i in Condition]


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
    ITIs=np.random.randint(ITI[0], ITI[1], size=Trials*6+1)
    # first ITI is placed after warmup, before first trial



    # Trials are TrialDuration ms long and first trial starts after warmup
    #Trial_end = np.cumsum(ITIs) + np.cumsum(np.ones(Trials)*TrialDuration)
    #Trial_start = Trial_end - TrialDuration
    #params['simtime'] = Trial_end[-1] + 200.

    StimulusDict, Clusters, StimTimes, StimAmplitudes = generateStimulusProtocol_Testing(
        6, Trials, ITIs, 1000, 500, PS_amplitude, PS_amplitude, [0, 1, 2, 3, 4, 5], Condition)

    params['multi_stim_clusters'] = Clusters
    params['multi_stim_times'] = StimTimes
    params['multi_stim_amps'] = StimAmplitudes

    params['stim_clusters_delay'] = [200.0]

    params['simtime'] = np.max([np.max(x) for x in StimTimes])

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST(default, params)
    EI_Network.setup_network()
    EI_Network.simulate()
    EI_Network.clean_up()

    # Analyze data, we are interested in the average firing rate of the excitatory and inhibitory population,
    # the average Fano factor of excitatory and inhibitory population and the average CV2 of the excitatory and
    # inhibitory population. We can sample 10 % of the neurons to get a good estimate of the population average.
    # But we need to make sure that we sample at least 10 neurons and sample the same neuron across jobs.

    np.random.seed(1)
    # get the neuron ids of the excitatory and inhibitory population
    N_E = params['N_E']
    N_I = params['N_I']
   # sample 20 % of the neurons per population
    N_E_sample = max(int(N_E /params['Q']/ 5), 10)
    N_I_sample = max(int(N_I / params['Q']/ 5), 10)

    # sample neurons
    E_sample = np.concatenate([np.random.choice(N_E//params['Q'], N_E_sample, replace=False)+ii*N_E/params['Q'] for ii in range(params['Q'])])
    I_sample = np.concatenate([np.random.choice(N_I//params['Q'], N_I_sample, replace=False)+ii*N_I/params['Q'] for ii in range(params['Q'])])

    # get the spike trains of the sampled neurons
    Spikes = EI_Network.get_recordings()
    #print(EI_Network.get_firing_rates(Spikes))
    E_Spikes_all = ExtractMeasurement.filterSpikes(Spikes, IDs=E_sample)
    I_Spikes_all = ExtractMeasurement.filterSpikes(Spikes, IDs=I_sample)
    # if there are no spikes, we can not estimate the Fano factor and CV2, so we return nan

    if os.getenv('SHOW_Plot', 'False')=='True':
        import matplotlib.pyplot as plt
        # plot spikes in range 3000 to 8000 ms
        index = np.logical_and(Spikes[0] > 3000, Spikes[0] < 8000)
        plt.figure()
        plt.plot(Spikes[0][index], Spikes[1][index], '.', color='black', label='_nolegend_', markersize=0.5)
        #plt.plot(Spikes[0], Spikes[1], '.', color='black', label='_nolegend_', markersize=0.5)
        plt.xlim(3000, 8000)
        plt.xlabel('Time (ms)')
        index = np.logical_and(StimulusDict['Marker'][:, 0] > 3000, StimulusDict['Marker'][:, 0] < 8000)
        for Mark in StimulusDict['Marker'][index]:
            plt.axvline(Mark[0], color='g')
            #plt.axvline(Mark[0] + StimulusDict['PreperatoryDuration'], color='r')
            #plt.axvline(Mark[0] + StimulusDict['PreperatoryDuration'] + StimulusDict['StimulusDuration'], color='blue')
        plt.show()
        plt.close()

    del Spikes

    window = [-500, 1000]
    maxWindow = 400

    # we have to estimate measurements independently for different combinations from direction and condition

    UsedCombination = np.unique([tuple(x) for x in StimulusDict['Marker'][:, 1:3]], axis=0)
    E_Spikes = []
    I_Spikes = []
    for ii in UsedCombination:
        # construct the trial start and end times for the current combination
        index = np.logical_and(StimulusDict['Marker'][:, 1] == ii[0], StimulusDict['Marker'][:, 2] == ii[1])
        Trial_start = StimulusDict['Marker'][index, 0] - maxWindow//2+window[0]
        Trial_end = StimulusDict['Marker'][index, 0] + maxWindow//2+window[1]
        E_Spikes.append(ExtractMeasurement.CutTrials(E_Spikes_all, Trial_start, Trial_end, TriggerTime=-window[0]+maxWindow//2))
        I_Spikes.append(ExtractMeasurement.CutTrials(I_Spikes_all, Trial_start, Trial_end, TriggerTime=-window[0]+maxWindow//2))

    FR_E_stim = []
    #FR_I_stim = []
    FR_E_nonstim = []
    #FR_I_nonstim = []
    FanoFactor_E_stim = []
    FanoFactor_E_nonstim = []
    CV2_E_stim = []
    CV2_E_nonstim = []

    Time_FR = []
    Time_Fano = []
    Time_CV2 = []

    for ii in range(len(E_Spikes)):
        direction = UsedCombination[ii][0]
        E_Spikes_stim = np.logical_and(E_Spikes[ii][1] > direction*N_E//params['Q'], E_Spikes[ii][1] < (direction+1)*N_E//params['Q'])
        E_Spikes_nonstim = np.logical_not(E_Spikes_stim)
        E_Spikes_stim = E_Spikes[ii][:, E_Spikes_stim]
        E_spike_IDs = np.unique(E_Spikes_stim[1])
        # replace the IDs in E_Spikes_stim with numbers from 0 to N_E_sample
        #for jj, ID in enumerate(E_spike_IDs):
        #    E_Spikes_stim[1][E_Spikes_stim[1] == ID] = jj
        E_Spikes_nonstim = E_Spikes[ii][:, E_Spikes_nonstim]
        #E_spike_IDs = np.unique(E_Spikes_nonstim[1])
        # replace the IDs in E_Spikes_nonstim with numbers from 0 to (Q-1)*N_E_sample
        #for jj, ID in enumerate(E_spike_IDs):
        #    E_Spikes_nonstim[1][E_Spikes_nonstim[1] == ID] = jj

        FR_loc, time_FR_loc = spiketools.kernel_rate(E_Spikes_stim[np.array([0,2])], tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2], kernel=spiketools.gaussian_kernel(50), dt=1, pool=True)
        FR_E_stim.append(FR_loc/N_E_sample)
        Time_FR.append(time_FR_loc)
        FR_loc, time_FR_loc = spiketools.kernel_rate(E_Spikes_nonstim[np.array([0,2])], tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2], kernel=spiketools.gaussian_kernel(50), dt=1, pool=True)
        FR_E_nonstim.append(FR_loc/(N_E_sample*(params['Q']-1)))
        Time_FR.append(time_FR_loc)
        for jj in np.unique(E_Spikes_stim[1]):
            E_spikes_stim_single = E_Spikes_stim[:, E_Spikes_stim[1] == jj]

            Fano_loc, time_Fano_loc = spiketools.kernel_fano(E_spikes_stim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2], dt=1)
            FanoFactor_E_stim.append(Fano_loc)
            Time_Fano.append(time_Fano_loc)

            CV_two_loc, time_CV_two_loc = spiketools.time_resolved_cv_two(E_spikes_stim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2])
            CV2_E_stim.append(CV_two_loc)
            Time_CV2.append(time_CV_two_loc)
        for jj in np.unique(E_Spikes_nonstim[1]):
            E_spikes_nonstim_single = E_Spikes_nonstim[:, E_Spikes_nonstim[1] == jj]

            Fano_loc, time_Fano_loc = spiketools.kernel_fano(E_spikes_nonstim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2], dt=1)
            FanoFactor_E_nonstim.append(Fano_loc)
            Time_Fano.append(time_Fano_loc)

            CV_two_loc, time_CV_two_loc = spiketools.time_resolved_cv_two(E_spikes_nonstim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2])
            CV2_E_nonstim.append(CV_two_loc)
            Time_CV2.append(time_CV_two_loc)



    # FanoFactor_E_stim = []
    # FanoFactor_E_nonstim = []
    # CV2_E_stim = []
    # CV2_E_nonstim = []
    # for ii in range(len(E_Spikes)):
    #     direction = UsedCombination[ii][0]
    #     E_Spikes_stim = np.logical_and(E_Spikes[ii][1] > direction*N_E//params['Q'], E_Spikes[ii][1] < (direction+1)*N_E//params['Q'])
    #     E_Spikes_stim= E_Spikes[ii][:, E_Spikes_stim]
    #     for jj in np.unique(E_Spikes_stim[1]):
    #         E_spikes_stim_single = E_Spikes_stim[:, E_Spikes_stim[1] == jj]
    #         FanoFactor_E_stim.append(spiketools.kernel_fano(E_spikes_stim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2], dt=1)[0])
    #         CV2_E_stim.append(spiketools.time_resolved_cv_two(E_spikes_stim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2])[0])
    #
    #     E_Spikes_nonstim = np.logical_not(E_Spikes_stim)
    #     E_Spikes_nonstim = E_Spikes[ii][:, E_Spikes_nonstim]
    #     for jj in np.unique(E_Spikes_nonstim[1]):
    #         E_spikes_nonstim_single = E_Spikes_nonstim[:, E_Spikes_nonstim[1] == jj]
    #         FanoFactor_E_nonstim.append(spiketools.kernel_fano(E_spikes_nonstim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2], dt=1)[0])
    #         CV2_E_nonstim.append(spiketools.time_resolved_cv_two(E_spikes_nonstim_single[[0,2]], window=400, tlim=[window[0]-maxWindow//2, window[1]+maxWindow//2])[0])

    if os.getenv('SHOW_Plot', 'False')=='True':
        fig, ax =plt.subplots(3,1, sharex=True)
        for ii in range(len(FR_E_stim)):
            ax[0].plot(FR_E_stim[ii].T, label='Stim')
            ax[0].plot(FR_E_nonstim[ii].T, label='NonStim')
        ax[0].legend()
        ax[0].set_title('Firing Rate')
        ax[1].plot(np.nanmean(FanoFactor_E_stim, axis=0), label='Stim')
        ax[1].plot(np.nanmean(FanoFactor_E_nonstim, axis=0), label='NonStim')
        ax[1].legend()
        ax[1].set_title('Fano Factor')
        ax[2].plot(np.nanmean(CV2_E_stim, axis=0), label='Stim')
        ax[2].plot(np.nanmean(CV2_E_nonstim, axis=0), label='NonStim')
        ax[2].legend()
        ax[2].set_title('CV2')
        plt.show()

    # measure the weighted sum of squared errors between the experimental data and the simulation data
    # we also want to find the time shift which minimizes the error
    # the experimental data is a dictionary which contains average timeresolved firing rate, fanofactor and cv_two and
    # the corresponding time points and the parameters of the estimators

    #MeanFF = calculate_mean_measure(np.nanmean(FanoFactor_E_stim, axis=0), np.nanmean(FanoFactor_E_nonstim, axis=0), 6)
    #MeanFR = calculate_mean_measure(np.nanmean(FR_E_stim, axis=0), np.nanmean(FR_E_nonstim, axis=0), 6)
    #MeanCV2 = calculate_mean_measure(np.nanmean(CV2_E_stim, axis=0), np.nanmean(CV2_E_nonstim, axis=0), 6)

    # without time shift
    #L2_FF = calculate_weighted_L2_distance(Experimental['FanoFactor'], MeanFF, np.ones_like(Experimental['FanoFactor']))
    #L2_FR = calculate_weighted_L2_distance(Experimental['FiringRate'], MeanFR, np.ones_like(Experimental['FiringRate']))
    #L2_CV2 = calculate_weighted_L2_distance(Experimental['CV2'], MeanCV2, np.ones_like(Experimental['CV2']))

    #L2_FF_stim = calculate_weighted_L2_distance(Experimental['FanoFactor'], np.nanmean(FanoFactor_E_stim, axis=0), np.ones_like(Experimental['FanoFactor']))
    #L2_FF_nonstim = calculate_weighted_L2_distance(Experimental['FanoFactor'], np.nanmean(FanoFactor_E_nonstim, axis=0), np.ones_like(Experimental['FanoFactor']))
    #L2_FR_stim = calculate_weighted_L2_distance(Experimental['FiringRate'], np.nanmean(FR_E_stim, axis=0), np.ones_like(Experimental['FiringRate']))
    #L2_FR_nonstim = calculate_weighted_L2_distance(Experimental['FiringRate'], np.nanmean(FR_E_nonstim, axis=0), np.ones_like(Experimental['FiringRate']))
    #L2_CV2_stim = calculate_weighted_L2_distance(Experimental['CV2'], np.nanmean(CV2_E_stim, axis=0), np.ones_like(Experimental['CV2']))
    #L2_CV2_nonstim = calculate_weighted_L2_distance(Experimental['CV2'], np.nanmean(CV2_E_nonstim, axis=0), np.ones_like(Experimental['CV2']))

    # with time shift
    #maxShift = 50 #ms
    #L2_CV2_TS = np.inf*np.ones(maxShift)
    #L2_FF_TS = np.inf*np.ones(maxShift)
    #L2_FR_TS = np.inf*np.ones(maxShift)

    #for ts in range(0, maxShift):
    #    L2_FF_TS[ts] = calculate_weighted_L2_distance_Timeshift(Experimental['FanoFactor'], MeanFF, np.ones_like(Experimental['FanoFactor']), ts)
    #    L2_FR_TS[ts] = calculate_weighted_L2_distance_Timeshift(Experimental['FiringRate'], MeanFR, np.ones_like(Experimental['FiringRate']), ts)
    #    L2_CV2_TS[ts] = calculate_weighted_L2_distance_Timeshift(Experimental['CV2'], MeanCV2, np.ones_like(Experimental['CV2']), ts)

    # find the time shift which minimizes the error for a*FF+b*FR+c*CV2
    # Weighting = np.array([1, 1, 1])
    # OverallError = np.vstack([L2_FF_TS, L2_FR_TS, L2_CV2_TS]).T@Weighting
    # minIndex = np.argmin(OverallError)
    # minError = OverallError[minIndex]
    # minShift = minIndex


    #with open(output_path, 'wb') as outfile:
    #    pickle.dump(EI_Network.get_recordings(), outfile)
    #    pickle.dump(EI_Network.get_parameter(), outfile)
    #    pickle.dump(StimulusDict, outfile)
    #    pickle.dump({'gitHash': gitHash, 'JobID': JobID, 'ArrayID': ArrayID}, outfile)


    FR_E_stim = np.nanmean(FR_E_stim, axis=0)
    FR_E_nonstim = np.nanmean(FR_E_nonstim, axis=0)
    FanoFactor_E_stim = np.nanmean(FanoFactor_E_stim, axis=0)
    FanoFactor_E_nonstim = np.nanmean(FanoFactor_E_nonstim, axis=0)
    CV2_E_stim = np.nanmean(CV2_E_stim, axis=0)
    CV2_E_nonstim = np.nanmean(CV2_E_nonstim, axis=0)

    # check if Time_FR contains the same time points
    if not check_same_vectors(Time_FR):
        raise ValueError('Time_FR does not contain the same time points')
    else:
        Time_FR = Time_FR[0]

    if not check_same_vectors(Time_Fano):
        raise ValueError('Time_Fano does not contain the same time points')
    else:
        Time_Fano = Time_Fano[0]

    if not check_same_vectors(Time_CV2):
        raise ValueError('Time_CV2 does not contain the same time points')
    else:
        Time_CV2 = Time_CV2[0]


    ResultDict={
        'FR_E_stim': FR_E_stim, 'FR_E_nonstim': FR_E_nonstim,
        'Time_FR': Time_FR,
        'FanoFactor_E_stim': FanoFactor_E_stim, 'FanoFactor_E_nonstim': FanoFactor_E_nonstim,
        'Time_Fano': Time_Fano,
        'CV2_E_stim': CV2_E_stim, 'CV2_E_nonstim': CV2_E_nonstim,
        'Time_CV2': Time_CV2,
        #'E_FiringRate': np.nanmean(E_FiringRate), 'I_FiringRate': np.nanmean(I_FiringRate), 'E_FanoFactor': np.nanmean(E_FanoFactor),
        #             'I_FanoFactor': np.nanmean(I_FanoFactor), 'CV2_E': np.nanmean(CV2_E), 'CV2_I': np.nanmean(CV2_I),
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

    #End = time.time()
    #print('Time: ', End-Start)
    # return exit code 0 if everything went fine
    sys.exit(0)


#def data_analysis(spiketimes, experimental_fit):
    # the experimental fit is a dictionary which contains average timeresolved firing rate, fanofactor and cv_two and
    # the corresponding time points and the parameters of the estimators
    # we want to compare the experimental data with the simulation data with a weighted sum of squared errors
    # there might be a  time shift between the experimental data and the simulation data, so we want to find the time shift
    # which minimizes the error

    
