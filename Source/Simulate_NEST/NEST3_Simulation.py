import numpy as np
import pickle
import sys
import time

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelNEST
import psutil

if __name__ == '__main__':

    FactorSize = 1
    FactorTime = 1
    Savepath = "Data.pkl"

    if len(sys.argv) == 2:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[1])

    elif len(sys.argv) == 3:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
    elif len(sys.argv) == 4:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
    elif len(sys.argv) == 5:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
        Savepath = sys.argv[4]

    elif len(sys.argv) >= 6:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
        Savepath = sys.argv[4]
        print("Too many arguments")

    print("FactorSize: " + str(FactorSize) + " FactorTime: " + str(FactorTime))

    CPUcount=psutil.cpu_count(logical = False)
    if CPUcount>8:
        CPUcount-=2

    startTime = time.time()
    baseline = {'N_E': 80, 'N_I': 20,  # number of E/I neurons -> typical 4:1
                'simtime': 900, 'warmup': 100}

    params = {'n_jobs': CPUcount, 'N_E': FactorSize * baseline['N_E'], 'N_I': FactorSize * baseline['N_I'], 'dt': 0.1,
              'neuron_type': 'iaf_psc_exp', 'simtime': FactorTime * baseline['simtime'], 'delta_I_xE': 0.,
              'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': FactorTime * baseline['warmup'],
              'Q': 20}

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 10.0  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])

    I_ths = [2.13,
             1.24]  # 3,5,Hz        #background stimulation of E/I neurons -> sets firing rates and changes behavior
    # to some degree # I_ths = [5.34,2.61] # 10,15,Hzh

    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]
    timeout = 18000  # 5h

    params['matrixType'] = "PROCEDURAL_GLOBALG"  # not needed only added for easier plotting scripts

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST_Timing(default, params)
    # Creates object which creates the EI clustered network in NEST
    Result = EI_Network.get_simulation(timeout=timeout)
    stopTime = time.time()
    Result['Timing']['Total'] = stopTime - startTime
    print("Total time     : %.4f s" % Result['Timing']['Total'])
    del Result['spiketimes']
    print(Result)

    with open(Savepath, 'ab') as outfile:
        pickle.dump(Result, outfile)
