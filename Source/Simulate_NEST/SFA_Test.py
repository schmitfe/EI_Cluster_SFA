import numpy as np
import pickle
import sys
import os
from pathlib import Path
import hashlib

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelNEST

if __name__ == '__main__':
    #get SLURM environment variables
    CPUcount = int(os.environ.get('SLURM_CPUS_PER_TASK', '10'))
    JobID = os.environ.get('SLURM_JOB_ID', '0')
    ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    #get git hash
    gitHash = os.popen('git rev-parse HEAD').read().strip()

    #get enviroment variables for simulation parmaeters
    randseed = int(os.environ.get('randseed', str(np.random.randint(1000000))))
    # Jep, jip_ratio, I_th_E, I_th_I, N_E, tau_stc, q_stc
    jep = 11.5
    jip_ratio = float(os.environ.get('jip_ratio', '0.75'))
    I_th_E = 2.4
    I_th_I = 1.25
    N_E = 4000
    N_I = 1000
    tau_stc = 180.0
    q_stc = 10.0
    #get enviroment variables for output_path
    output_path = os.environ.get('output_path', '../output/')
    #get enviroment variables for simulation protocol
    #Trials per direction, ITI, Preperatory duration, Stimulus duration, Stimulus intensity, Stimulus type
    

    #calculate inhibitory weight
    jip = 1. + (jep - 1) * jip_ratio

    #test if output path ends with a directory seperator
    if output_path.endswith(os.path.sep):
        #create a short name for the output file based on the parameters hash
        output_file = 'SFA_Sample'
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
             'Q': 20, 'jplus': np.array([[jep, jip], [jip, jip]]), 'I_th_E': I_th_E, 'I_th_I': I_th_I,
             'warmup': 5000., 'tau_stc': tau_stc, 'q_stc': q_stc, 'randseed': randseed}

    params['multi_stim_clusters']=None
    params['simtime'] = 5000. 
    print(params)

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST(default, params)
    EI_Network.setup_network()
    EI_Network.simulate()

    # save data

    with open(output_path, 'wb') as outfile:
        pickle.dump(EI_Network.get_recordings(), outfile)
        pickle.dump(EI_Network.get_parameter(), outfile)
        #pickle.dump(StimulusDict, outfile)
        pickle.dump({'gitHash': gitHash, 'JobID': JobID, 'ArrayID': ArrayID}, outfile)
    EI_Network.clean_up()