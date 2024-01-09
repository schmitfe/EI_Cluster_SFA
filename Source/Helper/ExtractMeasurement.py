import numpy as np
import pickle
import glob
import sys
import os
import pandas as pd
import spiketools

def filterSpikes(spikesArg, time=None, ID_range=None, IDs=None):
    """
    Filter spikes between time and ID
    :param spikes: numpy array with spikes (row 0 time, row 1 ID)
    :param time: tuple with start and end time
    :param ID_range: tuple with start and end ID
    :param IDs: list with IDs
    :return: filtered spikes
    """
    spikes = spikesArg.copy()
    if time is not None:

        spikes = spikes[:,(spikes[0, :] >= time[0]) & (spikes[0, :] <= time[1])]
    if ID_range is not None:
        spikes = spikes[:,(spikes[1, :] >= ID[0]) & (spikes[1, :] <= ID[1])]
    elif IDs is not None:
        spikes = spikes[:,np.isin(spikes[1, :], IDs)]
    return spikes

def CutTrials(spikes, trialStart, trialEnd):
    """
    Cut spikes into trials
    :param spikes: numpy array with spikes (row 0 time, row 1 ID)
    :param trialTimes: numpy array with trial times (row 0 start time, row 1 end time)
    :return: numpy array with spikes (row 0 time, row 1 ID, row 2 trial number)
    """
    spikes = spikes.copy()
    trials = []
    for ii, (TS, TE) in enumerate(zip(trialStart, trialEnd)):
        FilteredSpikes = filterSpikes(spikes, time=(TS, TE))
        FilteredSpikes[0, :] = FilteredSpikes[0, :] - TS
        # add trial number to spikes
        FilteredSpikes = addTrialNumber(FilteredSpikes, ii)
        trials.append(FilteredSpikes)

    # concatenate trials
    spikes = np.hstack(trials)
    return spikes


# function which adds 3rd row to spikes array with trial number
def addTrialNumber(spikes, trialNumber):
    """
    Add trial number to spikes
    :param spikes: numpy array with spikes (row 0 time, row 1 ID)
    :param trialNumber: trial number
    :return: spikes with trial number
    """
    spikes = spikes.copy()
    spikes = np.vstack((spikes, np.ones(spikes.shape[1]) * trialNumber))
    return spikes

# function which applies a function to each neuron and returns the result in a list
#
def applyToNeurons(spikes, function, returnTime, *args, **kwargs):
    """
    Apply function to each neuron
    :param spikes: numpy array with spikes (row 0 time, row 1 ID)
    :param function: function to apply
    :param args: arguments for function
    :param kwargs: keyword arguments for function
    :return: list with results
    """
    spikes=np.array(spikes)
    results = []
    for neuron in np.unique(spikes[1, :]):
        spikesLoc=spikes[:,spikes[1, :]==neuron]
        result_Iteration=function(spikesLoc[[0,2]], *args, **kwargs)
        if len(result_Iteration) == 2 and returnTime:
            results.append(result_Iteration)
        else:
            results.append(result_Iteration[0])
    return results



if __name__ == '__main__':
    #optinal argument to specify the path to the input files
    #if no argument is given, the default path is used
    #if .pkl is given, only this file is used
    #if .txt is given, all files in the txt file are used
    #if a folder is given, all .pkl files in this folder are used

    #default path
    filepath = '../output'
    #get list of all output files
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if filepath.endswith('.pkl'):
            input_files = [filepath]
        elif filepath.endswith('.txt'):
            with open(filepath, 'r') as infile:
                input_files = [line.strip() for line in infile.readlines()]
        else:
            input_files = glob.glob(filepath + '/*.pkl')
    else:
        input_files = glob.glob(filepath + '/*.pkl')

    # optinal second argument to specify the path to the output folder
    # if no argument is given, the default path is used
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    else:
        output_folder = '../output/ExtractedData'

    #create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    #use environment variable to specify kernel width for firing rate estimation
    #if no environment variable is set, use default value
    if 'KernelWidth' in os.environ:
        KernelWidth = float(os.environ['KernelWidth'])
    else:
        KernelWidth = 50.0
    kernel = spiketools.gaussian_kernel(KernelWidth)
    #use environment variable to specify window size for Fano factor estimation
    #if no environment variable is set, use default value
    if 'WindowSize' in os.environ:
        WindowSize = float(os.environ['WindowSize'])
    else:
        WindowSize = 400.0
    #use environment variable to specify window size for CV2 estimation
    #if no environment variable is set, use default value
    if 'WindowSizeCV2' in os.environ:
        WindowSizeCV2 = float(os.environ['WindowSizeCV2'])
    else:
        WindowSizeCV2 = 400.0



    #loop over all files
    for input_file in input_files:
        #open file
        with open(input_file, 'rb') as outfile:
            spikes = pickle.load(outfile)
            params = pickle.load(outfile)
            StimulusDict = pickle.load(outfile)
            ExtraInformation = pickle.load(outfile)


        ExtractedData = {}
        # get the marker times
        Marker = StimulusDict['Marker']
        # get the preperatory duration
        PreperatoryDuration = StimulusDict['PreperatoryDuration']
        # get the stimulus duration
        StimulusDuration = StimulusDict['StimulusDuration']
        # get the ITI duration
        ITI = StimulusDict['ITI']

        preMark = max(0.5 * ITI, 0.5 * KernelWidth, 0.5 * WindowSize)   #Marker is at start of prep. Stimulus
        postMark = PreperatoryDuration+StimulusDuration + max(0.5 * ITI, 0.5 * KernelWidth, 0.5 * WindowSize)

        #get the number of N_E and N_I neurons and number of Clusters
        N_E = params['N_E']
        N_I = params['N_I']
        N_Q = params['Q']




        #cut spikes into trials and save them in a list
        #spikes are cut into trials according to the marker times
        # marker[:,0] - PreperatoryDuration - max(0.5*ITI, 0.5*kernelwidth, 0.5*windowsize)) to
        # marker[:,0] + StimulusDuration + max(0.5*ITI, 0.5*kernelwidth, 0.5*windowsize))

        StimulusOnset=[]
        Direction=[]
        Condition=[]
        Cluster=[]
        SpikeTimes=[]
        for ii in Marker:
            SpikesInTrial = filterSpikes(spikes, time=(ii[0] - preMark, ii[0] + postMark))
            for jj in range(2*N_Q):
                StimulusOnset.append(ii[0])
                Direction.append(ii[1])
                Condition.append(ii[2])
                if jj < N_Q:
                    Cluster.append("E"+str(jj))
                    SpikeTimes.append(filterSpikes(SpikesInTrial, ID=(jj*N_E/N_Q, (jj+1)*N_E/N_Q)))
                else:
                    Cluster.append("I"+str(jj-N_Q))
                    SpikeTimes.append(filterSpikes(SpikesInTrial, ID=(N_E+(jj-N_Q)*N_I/N_Q, N_E+((jj-N_Q)+1)*N_I/N_Q)))
                # substract the stimulus onset time from the spike times
                SpikeTimes[-1][0, :] = SpikeTimes[-1][0, :] - StimulusOnset[-1]

        dfDict={'Stimulus_Onset': StimulusOnset, 'Direction': Direction, 'Condition': Condition,
                'Cluster': Cluster, 'Spike_Times': SpikeTimes}

        df = pd.DataFrame(dfDict)
        #add trial numbers per direction, condition and cluster
        df['Trial'] = df.groupby(['Direction', 'Condition', 'Cluster']).cumcount()+1
        #estimate firing rate per cluster, condition and direction
        df['Firing_Rate'] = df.apply(lambda row: spiketools.kernel_rate(row.Spike_Times, tlim=[-preMark, postMark], kernel=kernel, dt=1, pool=True)[0][0], axis=1)
        #add trial numbers to spike times
        df['Spike_Times'] = df.apply(lambda row: addTrialNumber(row.Spike_Times, row.Trial), axis=1)
        #estimate Fano factor per cluster, condition and direction
        #df['Fano_Factor'] = df.apply(lambda row: spiketools.kernel_fano(WindowSize, dt=1, pool=True), axis=1)


        #estimate CV2 per cluster, condition and direction


        #group by cluster, condition and direction and calculate mean time-resolved firing rate
        df2=df.groupby(['Cluster', 'Condition', 'Direction']).agg(FiringRate = ('Firing_Rate', lambda x: np.vstack(x).mean(axis=0).tolist()),
                                                                    CV2=('Spike_Times', lambda x: applyToNeurons(np.hstack(x), spiketools.time_resolved_cv2, False, window=WindowSizeCV2, tlim=[-preMark, postMark], pool=True, tstep=25)),
                                                                    FanoFactor=('Spike_Times', lambda x: applyToNeurons(np.hstack(x), spiketools.kernel_fano, False, window=WindowSizeCV2, tlim=[-preMark, postMark], dt=25))
                                                                    )
        #import matplotlib.pyplot as plt
        #for ii, row in df2.iterrows():
        #    plt.plot(np.nanmean(row['CV2'], axis=0))
        #plt.show()



        #get the name of the file without the path
        filename = os.path.basename(input_file)
        #save the spikes and stimulus protocol in a pickle file
        with open('../output/ExtractedData/' + filename, 'wb') as outfile:
            pickle.dump(params, outfile)
            pickle.dump(StimulusDict, outfile)
            pickle.dump(ExtraInformation, outfile)
            pickle.dump(ExtractedData, outfile)


