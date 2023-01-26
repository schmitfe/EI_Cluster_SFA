import numpy as np
import pickle
import sys
import os
from pathlib import Path
import hashlib

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelNEST

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

    #iterate through stimulus protocol and generate stimulus times and amplitudes
    time=ITI
    for ii in range(StimulusProtocol.shape[0]):
        #get stimulus times and amplitudes
        if ii%2 == 0:
            #preperatory stimulus
            #insert for each cluster which is marked as True in the stimulus protocol the stimulus amplitude of PrepIntensity
            for jj in range(Q):
                if StimulusProtocol[ii, jj]:
                    StimAmplitudes[jj].append(PrepIntensity)
                    StimTimes[jj].append(time)
            Marker[ii//2,0] = time
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
            time+=StimulusDuration+ITI
    #convert to numpy arrays
    for jj in range(Q):
        StimTimes[jj] = np.array(StimTimes[jj])
        StimAmplitudes[jj] = np.array(StimAmplitudes[jj])
    Clusters= [[ii] for ii in range(Q)]
    StimulusDict= {'Marker': Marker, 'ITI': ITI, 'PreperatoryDuration': PreperatoryDuration,
                   'StimulusDuration': StimulusDuration, 'PrepIntensity': PrepIntensity,
                   'StimulusIntensity': StimulusIntensity}
    return StimulusDict, Clusters, StimTimes, StimAmplitudes



if __name__ == '__main__':
    #get SLURM environment variables
    CPUcount = int(os.environ.get('SLURM_CPUS_PER_TASK', '14'))
    JobID = os.environ.get('SLURM_JOB_ID', '0')
    ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    #get git hash
    gitHash = os.popen('git rev-parse HEAD').read().strip()

    #get enviroment variables for simulation parmaeters
    randseed = int(os.environ.get('randseed', str(np.random.randint(1000000))))
    # Jep, jip_ratio, I_th_E, I_th_I, N_E, tau_stc, q_stc
    jep = float(os.environ.get('Jep', str(default.jep)))
    jip_ratio = float(os.environ.get('jip_ratio', '0.75'))
    I_th_E = float(os.environ.get('I_th_E', str(default.I_th_E)))
    I_th_I = float(os.environ.get('I_th_I', str(default.I_th_I)))
    N_E = int(os.environ.get('N_E', str(default.N_E)))
    N_I = int(os.environ.get('N_I', str(default.N_I)))
    tau_stc = float(os.environ.get('tau_stc', str(default.tau_stc)))
    q_stc = float(os.environ.get('q_stc', str(default.q_stc)))
    #get enviroment variables for output_path
    output_path = os.environ.get('output_path', '../output/')
    #get enviroment variables for simulation protocol
    #Trials per direction, ITI, Preperatory duration, Stimulus duration, Stimulus intensity, Stimulus type
    TrialsPerDirection = int(os.environ.get('TrialsPerDirection', '2'))
    ITI = float(os.environ.get('ITI', '500'))
    PreperatoryDuration = float(os.environ.get('PreperatoryDuration', '1000'))
    StimulusDuration = float(os.environ.get('StimulusDuration', '300'))
    PrepIntensity = float(os.environ.get('PrepIntensity', '0.5'))
    StimulusIntensity = float(os.environ.get('StimulusIntensity', '0.5'))

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
        output_file = 'jep_{}_jip_{}_IthE_{}_IthI_{}_NE_{}_NI_{}_tau_{}_q_{}_seed_{}_TrialsPerDirection_{}_ITI_{}_PrepDur_{}_StimDur_{}_PrepInt_{}_StimInt_{}'.format(jep, jip, I_th_E, I_th_I, N_E, N_I, tau_stc, q_stc, randseed, TrialsPerDirection, ITI, PreperatoryDuration, StimulusDuration, PrepIntensity, StimulusIntensity)
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
    if os.path.isfile(output_path):
        raise ValueError('Output file already exists')



    params= {'n_jobs': CPUcount, 'N_E': N_E, 'N_I': N_I, 'dt': 0.1, 'neuron_type': 'gif_psc_exp',
             'Q': 6, 'jplus': np.array([[jep, jip], [jip, jip]]), 'I_th_E': I_th_E, 'I_th_I': I_th_I,
             'warmup': 1000., 'tau_stc': tau_stc, 'q_stc': q_stc, 'randseed': randseed}

    # create stimulus protocol
    StimulusDict, Clusters, StimTimes, StimAmplitudes = generateStimulusProtocol_Testing(params['Q'],
                                        TrialsPerDirection, ITI, PreperatoryDuration , StimulusDuration, PrepIntensity,
                                                                                    StimulusIntensity, [2], [1,2,3])

    params['multi_stim_clusters'] = Clusters
    params['multi_stim_times'] = StimTimes
    params['multi_stim_amps'] = StimAmplitudes

    params['simtime'] = np.max([np.max(x) for x in StimTimes])

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST(default, params)
    EI_Network.setup_network()
    EI_Network.simulate()

    # save data

    with open(output_path, 'wb') as outfile:
        pickle.dump(EI_Network.get_recordings(), outfile)
        pickle.dump(EI_Network.get_parameter(), outfile)
        pickle.dump(StimulusDict, outfile)
        pickle.dump({'gitHash': gitHash, 'JobID': JobID, 'ArrayID': ArrayID}, outfile)
    EI_Network.clean_up()