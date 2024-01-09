import pickle
import matplotlib.pyplot as plt
import numpy as np

#function which opens the output file and plots the spikes and stimulus protocol
def plotExperiment (output_path, dpi=320, ax=None, labels=True):
    with open(output_path, 'rb') as outfile:
        spikes = pickle.load(outfile)
        _ = pickle.load(outfile)
        StimulusDict = pickle.load(outfile)
        _ = pickle.load(outfile)
    Marker=StimulusDict['Marker']
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
        #dark  color of spikes
    ax.plot(spikes[0], spikes[1], '.',color='black', label='_nolegend_', markersize=0.5)
    for Mark in Marker:
        ax.axvline(Mark[0], color='g')
        ax.axvline(Mark[0]+StimulusDict['PreperatoryDuration'], color='r')
        ax.axvline(Mark[0]+StimulusDict['PreperatoryDuration']+StimulusDict['StimulusDuration'], color='blue')

    if labels:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Spike raster plot')
        ax.legend(['PS', 'RS', 'TE'])
    return ax



# main function to test the plot function
if __name__ == '__main__':
    #add path to search path for modules
    import sys
    sys.path.append('../Source')
    import matplotlib.pyplot as plt
    import pickle
    import glob

    #Get list of all output files
    output_files = glob.glob('../output/*.pkl')



