import numpy as np
import pickle
import sys
import time
import os

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelNEST


if __name__ == '__main__':
    #get SLURM environment variables
    CPUcount = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    JobID = os.environ.get('SLURM_JOB_ID', '0')
    ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    #get git hash
    gitHash = os.popen('git rev-parse HEAD').read().strip()

    #get enviroment variables for simulation parmaeters
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
    output_path = os.environ.get('output_path', 'Data.pkl')
    #get enviroment variables for simulation protocol
    #Trials per direction, ITI, Preperatory duration, Stimulus duration, Stimulus intensity, Stimulus type
    TrialsPerDirection = int(os.environ.get('TrialsPerDirection', '10'))
    ITI = float(os.environ.get('ITI', '0.5'))
    PreperatoryDuration = float(os.environ.get('PreperatoryDuration', '1.0'))
    StimulusDuration = float(os.environ.get('StimulusDuration', '1.0'))
    PrepIntensity = float(os.environ.get('PrepIntensity', '0.5'))
    StimulusIntensity = float(os.environ.get('StimulusIntensity', '0.5'))

    #check if N_I is a multiple of Q and N_E is 4 times N_I
    if N_I % default.Q != 0:
        raise ValueError('N_I must be a multiple of Q')
    if N_E != 4 * N_I:
        raise ValueError('N_E must be 4 times N_I')


    #calculate inhibitory weight
    jip = 1. + (jep - 1) * jip_ratio

    #create stimulus protocol


    params= {'n_jobs': CPUcount, 'N_E': N_E, 'N_I': N_I, 'dt': 0.1, 'neuron_type': 'gif_psc_exp',
             'Q': 6, 'jplus': np.array([[jep, jip], [jip, jip]]), 'I_th_E': I_th_E, 'I_th_I': I_th_I,
             'warmup': 1000., 'tau_stc': tau_stc, 'q_stc': q_stc}



    EI_Network = ClusterModelNEST.ClusteredNetworkNEST(default, params)
    EI_Network.setup_network()
    EI_Network.simulate()

    # save data
    #create output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'wb') as outfile:
        pickle.dump(EI_Network.get_recordings(), outfile)
        pickle.dump(EI_Network.get_parameters(), outfile)
        pickle.dump({'gitHash': gitHash, 'JobID': JobID, 'ArrayID': ArrayID}, outfile)
    EI_Network.Cleanup()
